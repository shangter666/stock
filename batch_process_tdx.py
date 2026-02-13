
import os
import struct
import pandas as pd
from datetime import datetime
from pytdx.reader import GbbqReader
import argparse
from concurrent.futures import ProcessPoolExecutor
import time

def read_day_file(file_path):
    records = []
    file_size = os.path.getsize(file_path)
    
    with open(file_path, 'rb') as f:
        while f.tell() < file_size:
            data = f.read(32)
            if len(data) < 32:
                break
            
            val = struct.unpack('<IIIIIfII', data)
            date_int = val[0]
            
            if date_int < 19800000 or date_int > 20991231:
                continue
                
            date_str = str(date_int)
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                continue

            rec = {
                'date': dt,
                'open': val[1] * 0.01,
                'high': val[2] * 0.01,
                'low': val[3] * 0.01,
                'close': val[4] * 0.01,
                'amount': val[5],
                'volume': val[6]
            }
            records.append(rec)
            
    return pd.DataFrame(records)

def calculate_forward_adj(df_day, events):
    """
    df_day: DataFrame with columns [date, open, high, low, close, amount, volume]
    events: DataFrame (pre-filtered for symbol) with gbbq columns
    """
    if df_day.empty:
        return df_day

    # Sort data descending
    df_day = df_day.sort_values('date', ascending=False).reset_index(drop=True)
    df_day['factor'] = 1.0
    
    if events.empty:
        df_day['adj_close'] = df_day['close']
        df_day['adj_open'] = df_day['open']
        df_day['adj_high'] = df_day['high']
        df_day['adj_low'] = df_day['low']
        return df_day.sort_values('date', ascending=True)

    # Ensure event dates are timestamps
    if events['datetime'].dtype != 'datetime64[ns]':
        events['datetime'] = pd.to_datetime(events['datetime'].astype(str), format='%Y%m%d', errors='coerce')
        
    events = events.sort_values('datetime', ascending=False)
    # Filter only relevant categories (1 = Ex-Rights/Dividend)
    events = events[events['category'] == 1]
    
    event_idx = 0
    factors = []
    current_factor = 1.0
    
    # Pre-extract event values to list of dicts for faster iteration
    event_list = events.to_dict('records')
    num_events = len(event_list)
    
    for idx, row in df_day.iterrows():
        while event_idx < num_events and event_list[event_idx]['datetime'] > row['date']:
            evt = event_list[event_idx]
            
            close_before = row['close'] # Approximate close before ex-date (using current row which is T-1 or earlier)
            
            cash = evt['hongli_panqianliutong'] / 10.0
            bonus = evt['songgu_qianzongguben'] / 10.0
            rights = evt['peigu_houzongguben'] / 10.0
            r_price = evt['peigujia_qianzongguben']
            
            p_ex = (close_before - cash + r_price * rights) / (1 + bonus + rights)
            
            if close_before > 0:
                step_factor = p_ex / close_before
            else:
                step_factor = 1.0
                
            current_factor *= step_factor
            event_idx += 1
            
        factors.append(current_factor)
        
    df_day['factor'] = factors
    
    # Calculate adjusted prices
    for col in ['open', 'high', 'low', 'close']:
        df_day[f'adj_{col}'] = df_day[col] * df_day['factor']
        
    return df_day.sort_values('date', ascending=True)

def process_file(args):
    """
    Worker function to process a single file.
    args: (file_path, output_path, events_subset)
    """
    file_path, output_path, events_subset = args
    
    try:
        df_day = read_day_file(file_path)
        if df_day.empty:
            return f"Skipped (Empty): {file_path}"
        
        df_adj = calculate_forward_adj(df_day, events_subset)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df_adj.to_csv(output_path, index=False, float_format='%.3f')
        return None # Success
    except Exception as e:
        return f"Error processing {file_path}: {e}"

def main():
    parser = argparse.ArgumentParser(description='Batch Convert TDX .day to Forward Adjusted CSV')
    parser.add_argument('--data-dir', type=str, default='data', help='Root directory of data (containing sh/sz)')
    parser.add_argument('--output-dir', type=str, default='data_csv', help='Output directory for CSVs')
    parser.add_argument('--gbbq', type=str, default='gbbq', help='Path to gbbq file')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--stocks-only', action='store_true', help='Only process A-share stocks (60x, 68x, 00x, 30x)')
    
    args = parser.parse_args()
    
    # 1. Read GBBQ
    print(f"Reading gbbq from {args.gbbq}...")
    reader = GbbqReader()
    try:
        df_gbbq = reader.get_df(args.gbbq)
        print(f"Loaded {len(df_gbbq)} events.")
    except Exception as e:
        print(f"Failed to read gbbq: {e}")
        return

    # Define Stock Prefixes
    # SH: 600, 601, 603, 605, 688, 689
    # SZ: 000, 001, 002, 003, 300, 301
    stock_prefixes = ['sh600', 'sh601', 'sh603', 'sh605', 'sh688', 'sh689',
                      'sz000', 'sz001', 'sz002', 'sz003', 'sz300', 'sz301']

    # 2. Collect files to process
    tasks = []
    print(f"Scanning {args.data_dir} for .day files...")
    
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith('.day'):
                file_path = os.path.join(root, file)
                filename = file.lower()
                
                # Check filter
                if args.stocks_only:
                    # Check against prefixes
                    is_stock = False
                    for p in stock_prefixes:
                        if filename.startswith(p):
                            is_stock = True
                            break
                    if not is_stock:
                        continue
                
                # Determine Market and Symbol
                # Usually path is .../sh/lday/sh600000.day
                # or just look at filename prefix
                filename = file.lower()
                if filename.startswith('sh'):
                    market_id = 1
                    symbol = filename[2:-4]
                elif filename.startswith('sz'):
                    market_id = 0
                    symbol = filename[2:-4]
                else:
                    # Fallback: guess from path
                    if '/sh/' in file_path.lower():
                        market_id = 1
                        symbol = filename[:-4]
                    elif '/sz/' in file_path.lower():
                        market_id = 0
                        symbol = filename[:-4]
                    else:
                        print(f"Unknown market for {file_path}, skipping.")
                        continue
                
                # Filter events for this stock
                # Passing the full dataframe to workers is inefficient if large.
                # Pass only relevant subset.
                events_subset = df_gbbq[(df_gbbq['code'] == symbol) & (df_gbbq['market'] == market_id)].copy()
                
                # Construct Output Path
                # mirror structure: data/sh/lday/sh000001.day -> data_csv/sh/lday/sh000001.csv
                rel_path = os.path.relpath(file_path, args.data_dir)
                output_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + '.csv')
                
                tasks.append((file_path, output_path, events_subset))

    print(f"Found {len(tasks)} files to process.")
    
    # 3. Process in parallel
    start_time = time.time()
    success_count = 0
    error_count = 0
    
    # Use ProcessPoolExecutor
    # Note: passing DataFrame across processes has overhead. 
    # But since subsets are small, it should be fine.
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_file, tasks))
        
    for res in results:
        if res is None:
            success_count += 1
        else:
            print(res)
            error_count += 1
            
    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.2f}s.")
    print(f"Success: {success_count}, Errors/Skipped: {error_count}")

if __name__ == "__main__":
    main()
