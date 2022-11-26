###############################################################
#                                                             #
# midi_util.py-- Contains utilities for working with MIDI     #
#                files                                        # 
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


###############################################################
# Project Modules                                             #
###############################################################


###############################################################
# Procedures                                                  #
###############################################################


###############################################################
#                                                             #
# PROCEDURE:                                                  #
#       read_midi                                             #
#                                                             #
# DESCRIPTION:                                                #
#       Read a midi file and extract the array of notes and   #
#       chords in the song                                    #
#                                                             #
###############################################################
def read_mid( midi_filename ):
	# Local variables
	midi_notes     = []
	notes_to_parse = None
	
	# Parse the midi file
	midi_file = music.converter.parse( midi_filename )

	# Separate the various instruments
	inst_part = music.instrument.partitionByInstrument( midi_file )

	# Extract the notes for the selected instrument	
	for part in inst_part.parts:
		if 'Piano' in str( part ):
			notes_to_parse = part.recurse()

			# Append the notes and cords from the midi file
			for item in notes_to_parse:

				# Item is a note
				if   ( isinstance( item, music.note.Note ) ):
					midi_notes.append( str(item.pitch) )

				# Item is a chord
				elif ( isinstance(item, music.chord.Chord ) ):
					midi_notes.append( '.'.join( str(n) for n in item.normalOrder ) )

	# Return array of notes ( numpy )
	return midi_notes 



###############################################################
# END OF FILE                                                 # 
###############################################################
