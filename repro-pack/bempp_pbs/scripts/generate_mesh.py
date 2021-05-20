import trimesh
import numpy
import os

nanoshaper_exec = sys.argv[1] # nanoshaper executable
config_file = sys.argv[2] # nanoshaper config

os.system(nanoshaper_exec + " " + config_file)
filename = 'triangulatedSurf'
faces = numpy.loadtxt(filename+'.face', dtype=int, skiprows=3, usecols=(0,1,2))
vertices = numpy.loadtxt(filename+'.vert', dtype=float, skiprows=3, usecols=(0,1,2))

print("Original mesh: %i elements"%len(faces))

mesh = trimesh.Trimesh(vertices = vertices, faces= faces-1)

mesh_split = mesh.split()
print("Found %i meshes"%len(mesh_split))

vertices_split = mesh_split[0].vertices
faces_split = mesh_split[0].faces
print("Mesh 1: %i elements"%len(faces_split))

numpy.savetxt(filename+'_split.face', faces_split+1, fmt='%i')
numpy.savetxt(filename+'_split.vert', vertices_split, fmt='%1.5f')