import hdf5_getters as getter
import numpy as np

f1 = '../data/MillionSongSubset/data/A/A/A/TRAAAAK128F9318786.h5'
f2 = '../data/MillionSongSubset/data/B/D/A/TRBDAAB12903CA6A79.h5'
h5 = getter.open_h5_file_read(f1)
h5_2 = getter.open_h5_file_read(f2)

print "Segments pitches"
print "F1 pitches shape", getter.get_segments_pitches(h5).shape
print "F2 pitches shape", getter.get_segments_pitches(h5_2).shape

np.savetxt('../data/pitch_sample.txt', getter.get_segments_pitches(h5))


print "Segments timbre"
np.savetxt('../data/timbre_sample.txt', getter.get_segments_timbre(h5))


print "F1 pitches shape", getter.get_segments_timbre(h5).shape
print "F2 pitches shape", getter.get_segments_timbre(h5_2).shape
