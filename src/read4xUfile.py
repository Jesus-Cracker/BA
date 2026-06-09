import numpy as np
from tqdm import tqdm
import csv

def read_4xU_file(file, csv_out=None):
    expected_packet_length = 76
    num_of_columns = 28
    
    with open(file, 'rb') as f:
        data = f.read().split(b'\x7E')

    data_len = len(data)

    time = np.zeros(data_len)
    subseconds = np.zeros(data_len)
    ecg1, ecg2, ecg3, ecg4 = (np.zeros(data_len,dtype=np.double) for _ in range(4))
    ppg1, ppg2, ppg3, ppg4 = (np.zeros(data_len) for _ in range(4))
    mi1, mi2, mi3, mi4 = (np.zeros(data_len) for _ in range(4))
    
    scg1, scg2, scg3, scg4 = (np.zeros((data_len, 3),dtype=np.int16) for _ in range(4))
    buf_fill_state = np.zeros(data_len)

    for i in tqdm(range(len(data))):
        data_char_array = bytearray(data[i])
        
        if len(data_char_array) > expected_packet_length:
            if data_char_array[6:8] == bytearray([0x00, 0x00]):  
                
                dest_data_array = data_char_array.copy()
                pos = 0
                j = 0
                while j < len(data_char_array):
                    if data_char_array[j] == 0x7D:
                        if j + 1 < len(data_char_array):
                            if data_char_array[j + 1] == 0x5D:
                                del dest_data_array[pos + 1]
                            elif data_char_array[j + 1] == 0x5E:
                                dest_data_array[pos] = 0x7E
                                del dest_data_array[pos + 1]
                        pos -= 1
                    pos += 1
                    j += 1
                
                data_dec_values = np.array(list(dest_data_array), dtype=np.float64)

                time[i] = sum(data_dec_values[0:8] * (2 ** np.arange(0, 64, 8)))
                subseconds[i] = sum(data_dec_values[72:74] * (2 ** np.arange(0, 16, 8))) / 1000
                time[i] += subseconds[i]

                ecg_values = np.array(data_dec_values[8:24],dtype=np.int32)      
                ecg1[i],ecg2[i],ecg3[i],ecg4[i]= [((ecg_values[3+4*k]<<24)+(ecg_values[2+4*k]<<16)+(ecg_values[1+4*k]<<8)+ecg_values[0+4*k])/pow(2,24)*4800 for k in range(4)]
                
                ppg_values = np.array([data_dec_values[28:30],data_dec_values[44:46],data_dec_values[60:62],data_dec_values[30:32]],dtype=np.int32)      
                ppg1[i],ppg2[i],ppg3[i],ppg4[i]= [((ppg_values[k,1]<<8)+ppg_values[k,0]) for k in range(4)]

                mim_values = np.array(data_dec_values[40:48],dtype=np.int32)      
                mi1[i],mi2[i],mi3[i],mi4[i]= [((mim_values[1+2*k]<<8)+mim_values[0+2*k]) for k in range(4)]

                scg_values = np.array([data_dec_values[np.array([49,48,51,50,53,52])] ,data_dec_values[np.array([55,54,57,56,59,58])],data_dec_values[np.array([61,60,63,62,65,64])],data_dec_values[np.array([67,66,69,68,71,70])]],dtype=np.int16)
                scg1[i],scg2[i],scg3[i],scg4[i]= [([(scg_values[k,0]<<8)+scg_values[k,1],(scg_values[k,2]<<8)+scg_values[k,3],(scg_values[k,4]<<8)+scg_values[k,5]]) for k in range(4)]
               
                buf_fill_state[i] = data_dec_values[74]

    
    idfs= np.where(time != 0)[0]
    
    dat_struct={
        "time": time[idfs],
        "ecg1": ecg1[idfs],
        "ecg2": ecg2[idfs],
        "ecg3": ecg3[idfs],
        "ecg4": ecg4[idfs],
        "ppg1": ppg1[idfs],
        "ppg2": ppg2[idfs],
        "ppg3": ppg3[idfs],
        "ppg4": ppg4[idfs],
        "mi1": mi1[idfs],
        "mi2": mi2[idfs],
        "mi3": mi3[idfs],
        "mi4": mi4[idfs],
        "scg1": scg1[idfs,:],
        "scg2": scg2[idfs,:],
        "scg3": scg3[idfs,:],
        "scg4": scg4[idfs,:],
        "bufFillState": buf_fill_state[idfs]
     }
    if csv_out is not None:
        with open(csv_out, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow([
                "time", "ecg1", "ecg2", "ecg3", "ecg4",
                "ppg1", "ppg2", "ppg3", "ppg4",
                "mi1", "mi2", "mi3", "mi4",
                "scg1_x", "scg1_y", "scg1_z",
                "scg2_x", "scg2_y", "scg2_z",
                "scg3_x", "scg3_y", "scg3_z",
                "scg4_x", "scg4_y", "scg4_z",
                "bufFillState"
            ])
            for i in range(len(dat_struct["time"])):
                writer.writerow([
                    dat_struct["time"][i],
                    dat_struct["ecg1"][i],
                    dat_struct["ecg2"][i],
                    dat_struct["ecg3"][i],
                    dat_struct["ecg4"][i],
                    dat_struct["ppg1"][i],
                    dat_struct["ppg2"][i],
                    dat_struct["ppg3"][i],
                    dat_struct["ppg4"][i],
                    dat_struct["mi1"][i],
                    dat_struct["mi2"][i],
                    dat_struct["mi3"][i],
                    dat_struct["mi4"][i],
                    *dat_struct["scg1"][i],
                    *dat_struct["scg2"][i],
                    *dat_struct["scg3"][i],
                    *dat_struct["scg4"][i],
                    dat_struct["bufFillState"][i]
                ])

    return dat_struct


def read_4xU_file_rt(data):
    expected_packet_length = 76
   
    data_len = int(len(data)/81)

    time = np.zeros(data_len)
    subseconds = np.zeros(data_len)
    ecg1, ecg2, ecg3, ecg4 = (np.zeros(data_len,dtype=np.double) for _ in range(4))
    ppg1, ppg2, ppg3, ppg4 = (np.zeros(data_len) for _ in range(4))
    mi1, mi2, mi3, mi4 = (np.zeros(data_len) for _ in range(4))
    
    scg1, scg2, scg3, scg4 = (np.zeros((data_len, 3),dtype=np.int16) for _ in range(4))
    buf_fill_state = np.zeros(data_len)

    for i in range(data_len):
        data_char_array = data[i*81:(i+1)*81]
        
        if len(data_char_array) > expected_packet_length:
            if data_char_array[6:8] == bytearray([0x00, 0x00]):  
                
                dest_data_array = data_char_array.copy()
                pos = 0
                j = 0
                while j < len(data_char_array):
                    if data_char_array[j] == 0x7D:
                        if j + 1 < len(data_char_array):
                            if data_char_array[j + 1] == 0x5D:
                                del dest_data_array[pos + 1]
                            elif data_char_array[j + 1] == 0x5E:
                                dest_data_array[pos] = 0x7E
                                del dest_data_array[pos + 1]
                        pos -= 1
                    pos += 1
                    j += 1
                
                data_dec_values = np.array(list(dest_data_array), dtype=np.float64)

                time[i] = sum(data_dec_values[0:8] * (2 ** np.arange(0, 64, 8)))
                subseconds[i] = sum(data_dec_values[72:74] * (2 ** np.arange(0, 16, 8))) / 1000
                time[i] += subseconds[i]

                ecg_values = np.array(data_dec_values[8:24],dtype=np.int32)      
                ecg1[i],ecg2[i],ecg3[i],ecg4[i]= [((ecg_values[3+4*k]<<24)+(ecg_values[2+4*k]<<16)+(ecg_values[1+4*k]<<8)+ecg_values[0+4*k])/pow(2,24)*4800 for k in range(4)]
                
                ppg_values = np.array([data_dec_values[28:30],data_dec_values[44:46],data_dec_values[60:62],data_dec_values[30:32]],dtype=np.int32)      
                ppg1[i],ppg2[i],ppg3[i],ppg4[i]= [((ppg_values[k,1]<<8)+ppg_values[k,0]) for k in range(4)]

                mim_values = np.array(data_dec_values[40:48],dtype=np.int32)      
                mi1[i],mi2[i],mi3[i],mi4[i]= [((mim_values[1+2*k]<<8)+mim_values[0+2*k]) for k in range(4)]

                scg_values = np.array([data_dec_values[np.array([49,48,51,50,53,52])] ,data_dec_values[np.array([55,54,57,56,59,58])],data_dec_values[np.array([61,60,63,62,65,64])],data_dec_values[np.array([67,66,69,68,71,70])]],dtype=np.int16)
                scg1[i],scg2[i],scg3[i],scg4[i]= [([(scg_values[k,0]<<8)+scg_values[k,1],(scg_values[k,2]<<8)+scg_values[k,3],(scg_values[k,4]<<8)+scg_values[k,5]]) for k in range(4)]
               
                buf_fill_state[i] = data_dec_values[74]

    
    idfs= np.where(time != 0)[0]
    
    data_struct={
        "time": time[idfs],
        "ecg1": ecg1[idfs],
        "ecg2": ecg2[idfs],
        "ecg3": ecg3[idfs],
        "ecg4": ecg4[idfs],
        "ppg1": ppg1[idfs],
        "ppg2": ppg2[idfs],
        "ppg3": ppg3[idfs],
        "ppg4": ppg4[idfs],
        "mi1": mi1[idfs],
        "mi2": mi2[idfs],
        "mi3": mi3[idfs],
        "mi4": mi4[idfs],
        "scg1": scg1[idfs,2],
        "scg2": scg2[idfs,2],
        "scg3": scg3[idfs,2],
        "scg4": scg4[idfs,2],
        "bufFillState": buf_fill_state[idfs]
     }
  
    return data_struct


