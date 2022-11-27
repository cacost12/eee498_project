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
#                                                             #
# PROCEDURE:                                                  #
#       export_midi                                           #
#                                                             #
# DESCRIPTION:                                                #
#       Export a series of notes to a midi file               #
#                                                             #
###############################################################
def export_midi( notes_list, output_filename ):
	
	# Local variables
	offset       = 0
	output_notes = []
	output_file  = 'output/' + output_filename + ".mid" 

	# Create note and chord objects 
	for item in notes_list:
		
		# Chord
		if ( '.' in item ) or item.isdigit():
			notes_in_chord   = item.split( '.' ) 
			midi_chord_notes = []
			# Convert each note in the chord to a note object
			for chord_note in notes_in_chord:
				chord_note_int = int( chord_note )
				new_note       = music.note.Note( chord_note_int )
				new_note.storedInstrument = music.instrument.Piano()
				midi_chord_notes.append( new_note )

			# Construct the chord from the notes
			new_chord = music.chord.Chord( midi_chord_notes )
			new_chord.offset = offset
			output_notes.append( new_chord )

		# Note
		else:
			new_note        = music.note.Note( item )
			new_note.offset = offset
			new_note.storedInstrument = music.instrument.Piano()
			output_notes.append( new_note )

		# Increment offset
		offset += 1

	# Output Data to file
	midi_fstream = music.stream.Stream( output_notes )		
	midi_fstream.write( "midi", fp= output_file )


###############################################################
# END OF FILE                                                 # 
###############################################################
