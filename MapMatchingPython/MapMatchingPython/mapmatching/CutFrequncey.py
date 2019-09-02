import pandas as pd
def cut_samplerate(gps_track,sample_rate):

    Len=len(gps_track)/sample_rate
    last_len=len(gps_track)%sample_rate
    delete_list=[]
    for i in range(Len):
        
        if i==Len-1:
            if last_len==0:
                for x in range(1,sample_rate-1):
                    delete_list.append(sample_rate*i+x)
            else:
                for x in range(1,sample_rate):
                    delete_list.append(sample_rate*i+x)
             
        else:
            for x in range(1,sample_rate):
                    delete_list.append(sample_rate*i+x)
         
       
          
    gps_track.drop(gps_track.index[delete_list],inplace=True)
    return gps_track



if __name__ == '__main__':
    gps_track=[0,1,2,3,4,5,6,7,8,9,10,11]
    gps_track=pd.DataFrame(gps_track)
    print gps_track

    print cut_samplerate(gps_track, 3)