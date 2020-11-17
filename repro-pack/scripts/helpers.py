import numpy as _np

def parse_pqr(pqr_file):
    """
    Parse pqr file and return charges and their coordinates.
    """
    charges = list()
    coords = list()
    with open(pqr_file, 'r') as f: 
        for line in f:
            line = line.strip().split()
            if len(line) > 0:
                if line[0] == 'ATOM':
                    coords.append(line[-5:-2])
                    charges.append(line[-2])
    charges = _np.array(charges, dtype=float)
    coords = _np.array(coords, dtype=float).reshape(-1,3)

    return charges, coords

def generate_grid(face_file, vert_file):
    """
    Create bempp Grid object from .face and .vert files.
    """
    import bempp.api
    face = open(face_file, 'r').read()
    vert = open(vert_file, 'r').read()
    faces = _np.vstack(_np.char.split(face.split('\n')[0:-1]))[:,:3].astype(int) - 1
    verts = _np.vstack(_np.char.split(vert.split('\n')[0:-1]))[:,:3].astype(float)
    grid = bempp.api.Grid(verts.transpose(), faces.transpose())

    return grid
