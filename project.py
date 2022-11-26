###############################################################
#                                                             #
# project2.py-- EEE498 Machine Learning with Deployment to    #
#               FPGA Final Project                            # 
#                                                             #
# Author: Colton Acosta                                       #
# Date: 11/26/2022                                            #
#                                                             #
###############################################################


###############################################################
# Standard Imports                                            #
###############################################################
import music21 as music
import numpy   as np
from   matplotlib  import pyplot as plt
from   collections import Counter
from   sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K


###############################################################
# Project Modules                                             #
###############################################################
import midi_util


###############################################################
# Global Variables                                            #
###############################################################

# Midi files and data 
midi_path = 'mid/'
midi_files = [ 
			  midi_path + "greendayrhythm.mid",
              midi_path + "greendayvocal.mid" , 
              midi_path + "hotellead.mid"     , 
              midi_path + "hotelrhythm.mid"   , 
              midi_path + "queenchorus.mid"   , 
              midi_path + "queenrhythm.mid"   , 
              midi_path + "queenvocal.mid"    , 
              midi_path + "vivalead.mid"      , 
              midi_path + "vivarythm.mid"       
             ]
midi_notes_list = []

# Model hyperparameters
note_freq_threshold  = 10  # only use notes occuring more than 10 times
num_timesteps        = 32  # Number of timesteps per song
test_train_split_per = 0.3 # Percentage of test/train data
random_seed          = 23


###############################################################
# Procedures                                                  #
###############################################################


###############################################################
#                                                             #
# PROCEDURE:                                                  #
#       flatten_list                                          #
#                                                             #
# DESCRIPTION:                                                #
#       Reduces a multidimensional list to a 1D list          #
#                                                             #
###############################################################
def flatten_list( input_list ):
    new_list = []
    for i in input_list:
        for j in i:
            new_list.append(j)
    return new_list


###############################################################
# Load Dataset                                                #
###############################################################

# Import notes
print( "Importing midi data ..." )
for midi_file in midi_files:
	song_midi_notes_list = midi_util.read_mid( midi_file )
	if ( len( song_midi_notes_list ) ):
		midi_notes_list.append( midi_util.read_mid( midi_file ) )	

# Convert to numpy array
midi_notes_list1D = flatten_list( midi_notes_list )
midi_notes_array  = np.array( midi_notes_list1D )
print( "Midi data imported sucessfully" )

# Visualize the dataset  
note_frequencies = dict( Counter( midi_notes_list1D ) )
note_freq_nums = [ note_count for _,note_count in note_frequencies.items() ]
plt.hist( note_freq_nums )
plt.show()

# Determine the most frequent notes 
freq_note_list = []
for midi_note, count in note_frequencies.items():
	if ( count >= note_freq_threshold ):
		freq_note_list.append( midi_note )

# Filter the infrequent notes from the music
filtered_notes_list = []
for song in midi_notes_list:
	note_buffer = []
	for midi_note in song:
		if midi_note in freq_note_list:
			note_buffer.append( midi_note ) 
	filtered_notes_list.append( note_buffer )
filtered_notes_list1D = flatten_list( filtered_notes_list )

# Prepare the input and output sequences
X = []
y = []
for midi_notes in filtered_notes_list:
	for i in range( 0, len(midi_notes) - num_timesteps, 1 ):
		input_note_seq = midi_notes[i:i+ num_timesteps]
		output_note    = midi_notes[i+ num_timesteps]
		X.append ( input_note_seq )
		y.append( output_note    )
X = np.array( X )
y = np.array( y )

# Assign a unique integer to each note
note_int_X     = list( set( X.ravel() ) )
note_int_y     = list( set( y         ) )
note_int_dic_X = dict( (midi_note, num) for num, midi_note in enumerate(note_int_X) )
note_int_dic_y = dict( (midi_note, num) for num, midi_note in enumerate(note_int_y) )

# Convert the note sequential data to integer data
X_seq = []
for midi_notes in X:
	int_buffer = []
	for midi_note in midi_notes:
		int_buffer.append( note_int_dic_X[midi_note] )
	X_seq.append( int_buffer )
X_seq = np.array( X_seq )
y_seq = np.array( [note_int_dic_y[midi_note] for midi_note in y] )

# Split the test and train data
X_train, X_test, Y_train, y_test = train_test_split( X_seq,
	                                                 y_seq,
                                                     test_size = test_train_split_per, 
                                                     random_state = random_seed )


###############################################################
# Machine Learning Model LTSM                                 #
###############################################################

ML_model = Sequential()
	

###############################################################
# END OF FILE                                                 # 
###############################################################
