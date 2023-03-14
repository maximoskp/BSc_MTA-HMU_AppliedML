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
            self.name = filepath.split('.')[-2].split(os.sep)[-1]
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
