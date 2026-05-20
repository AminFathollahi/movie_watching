import pandas as pd
timing = pd.read_excel('/home/amin/Research/Representation/Movie/data/Setareh/Data/TOI_NEW.xlsx')
timing_df = []
for _,row in timing.iterrows():
    r = row['Run_ID'] 
    v= row['Video_ID']
    start = row['Start(Secs)'] 
    Duration = row['Duration(secs)']
    end = row ['Our_End(secs)']
    if r ==2:
        start += 900 
        end += 900
    if r ==3:
        start += 900 + 898
        end += 900 + 898
    if r ==4:
        start += 900 + 898 + 895
        end += 900 + 898 + 895
    timing_df.append({'video_id':v,'onset_sec':start,'end_sec':end, 'duration_sec':Duration})

timing_df = pd.DataFrame(timing_df)
timing_df.to_csv('/home/amin/Research/Representation/Movie/data/Setareh/Data/movie_timing.csv', index = False)