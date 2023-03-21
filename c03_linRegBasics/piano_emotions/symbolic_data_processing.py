import music21 as m21
import numpy as np
import os
import pandas as pd

class SymbolicInfo:
    
    metadata = None
    
    def __init__(self, filepath, metadatafile=None, logging=False):
        if logging:
            print('processing ', filepath)
        if metadatafile is not None and SymbolicInfo.metadata is None:
            SymbolicInfo.metadata = pd.read_csv( metadatafile )
        if filepath.split('.')[-1] in ['xml', 'mid', 'midi', 'mxl', 'musicxml']:
            self.name = filepath.split('.')[-2].split('/')[-1]
            if SymbolicInfo.metadata is not None:
                self.title = self.metadata[ self.metadata['ID'] == self.name ]['Title']
            self.stream = m21.converter.parse( filepath )
            self.flat = self.stream.flat.notes
            self.pcs = []
            self.pcp = np.zeros(12)
            self.make_pcp()
            self.estimate_tonality()
        else:
            print('bad format')
    # end __init__

    def make_pcs(self):
        for p in self.flat.pitches:
            self.pcs.append( p.midi%12 )
    # end make_pcp
    
    def make_pcp(self):
        self.make_pcs()
        for p in self.pcs:
            self.pcp[p] += 1
        if np.sum(self.pcp) != 0:
            self.pcp = self.pcp/np.sum(self.pcp)
    # end make_pcp
    
    def estimate_tonality(self):
        p = m21.analysis.discrete.KrumhanslSchmuckler()
        self.estimated_tonality = p.getSolution(self.stream)
# end class SymbolicInfo


class SymbolicEmotionInfo():
    
    metadata = None
    major_profile = m21.analysis.discrete.KrumhanslSchmuckler().getWeights('major')
    minor_profile = m21.analysis.discrete.KrumhanslSchmuckler().getWeights('minor')
    
    def __init__(self, filepath, metadatafile=None, keep_aux=False, logging=False):
        if logging:
            print('processing ', filepath)
        if metadatafile is not None and SymbolicInfo.metadata is None:
            SymbolicEmotionInfo.metadata = pd.read_csv( metadatafile )
        if filepath.split('.')[-1] in ['xml', 'mid', 'midi', 'mxl', 'musicxml']:
            self.filename = filepath.split('.')[-2].split('/')[-1]
            self.name = '_'.join(self.filename[3:].split('_')[:-1])
            if SymbolicEmotionInfo.metadata is not None:
                self.dominantQ = self.metadata[ self.metadata['songID'] == self.name ]['DominantQ'].iloc[0]
                self.isHappy = True if self.dominantQ == 1 or self.dominantQ == 4 else False
                self.isEnergetic = True if self.dominantQ == 1 or self.dominantQ == 2 else False
                q1 = self.metadata[ self.metadata['songID'] == self.name ]['num_Q1'].iloc[0]
                q2 = self.metadata[ self.metadata['songID'] == self.name ]['num_Q2'].iloc[0]
                q3 = self.metadata[ self.metadata['songID'] == self.name ]['num_Q3'].iloc[0]
                q4 = self.metadata[ self.metadata['songID'] == self.name ]['num_Q4'].iloc[0]
                self.valence = ((q1+q4) - (q2+q3))/( q1+q2+q3+q4 )
                self.arousal = ((q1+q2) - (q3+q4))/( q1+q2+q3+q4 )
            self.stream = m21.converter.parse( filepath )
            # self.features_dictionary = fcf.compute_features_of_m21score( self.stream )
            self.flat = self.stream.flat.notes
            self.pcs = []
            self.onsets = []
            self.ioi = [] # inter-onset interval
            self.durations = []
            self.make_pcs_and_rhythms()
            self.make_rpcp()
            self.make_rhythm_features()
            self.features = np.hstack( (self.rpcp, self.rhythm_features) )
            if not keep_aux:
                del self.stream
                del self.flat
                del self.pcp
                del self.onsets
                del self.ioi
                del self.durations
                del self.rhythm_features
                del self.estimated_tonality
        else:
            print('bad format')
    # end __init__
    
    def make_pcs_and_rhythms(self):
        for p in self.flat.pitches:
            self.pcs.append( p.midi%12 )
        for n in self.flat:
            if n.offset not in self.onsets:
                self.onsets.append(float(n.offset))
            self.durations.append(float(n.duration.quarterLength))
        self.onsets = np.array( self.onsets )
        self.ioi = np.diff( self.onsets )
        self.durations = np.array( self.durations )
    # end make_pcs_and_rhythms
    
    def make_pcp(self):
        for p in self.pcs:
            self.pcp[p] += 1
        if np.sum(self.pcp) != 0:
            self.pcp = self.pcp/np.sum(self.pcp)
    # end make_pcp
    
    def make_rpcp(self):
        self.pcp = np.zeros(12)
        self.make_pcp()
        self.estimated_tonality = self.tonality_from_pcp()
        self.rpcp = np.roll( self.pcp, -self.estimated_tonality['root'] )
    # end make_rpcp
    
    def make_rhythm_features(self):
        self.rhythm_features = np.zeros(5)
        
        self.rhythm_features[0] = np.mean( self.ioi )
        self.rhythm_features[1] = np.std( self.ioi )
        self.rhythm_features[2] = np.mean( self.durations )
        self.rhythm_features[3] = np.std( self.durations )
        self.rhythm_features[4] = np.max( self.onsets )/self.onsets.size
    # end make_rhythm_features
    
    def tonality_from_pcp( self ):
        major_corrs = np.zeros(12).astype(np.float32)
        minor_corrs = np.zeros(12).astype(np.float32)
        for i in range(12):
            major_corrs[i] = np.corrcoef( self.pcp, np.roll( 
                self.major_profile, i ) )[0][1]
            minor_corrs[i] = np.corrcoef( self.pcp, np.roll( 
                self.minor_profile, i ) )[0][1]
        major_max_idx = np.argmax( major_corrs )
        minor_max_idx = np.argmax( minor_corrs )
        major_max = np.max( major_corrs )
        minor_max = np.max( minor_corrs )
        if major_max > minor_max:
            return {'root': major_max_idx,
                    'mode': 'major',
                    'correlation': major_max}
        else:
            return {'root': minor_max_idx,
                    'mode': 'minor',
                    'correlation': minor_max}
    # end tonality_from_pcp
# end class SymbolicEmotionInfo
