from mayavi import mlab
import h5py

f = h5py.File('volve.vol', 'r')
s = f['seis'][:]
s = s.reshape(200, 200, 200)
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s), plane_orientation='x_axes', colormap='seismic')
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s), plane_orientation='y_axes', colormap='seismic')
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s), plane_orientation='z_axes', colormap='seismic')
mlab.orientation_axes()
mlab.show()

#How to install
# (venv) agus@agus-B250GT3:~/PycharmProjects/DSML_INTERMEDIATE
# git clone https://github.com/enthought/mayavi.git
# cd mayavi
# pip install -r requirements.txt
# pip install PyQt5
# python setup.py install
# pip install h5py