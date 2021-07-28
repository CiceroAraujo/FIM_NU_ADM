import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from ..inputs import finescale_inputs as inputs

class FieldVisualizer():
    def __init__(self):
        self.grid=self.get_grid()

    def plot_labels(self, values):
        self.grid.point_arrays["values"] = values.flatten().astype(np.float64)  # Flatten the array!
        plotter = pv.Plotter()
        plotter.add_mesh(self.grid, show_edges=True, color="tan")
        points = self.grid.points
        mask = points[:, 0] == 0
        plotter.add_point_labels(points, values.round(3).tolist(), point_size=20, font_size=15)
        plotter.show()


    def plot_field(self, values):
        self.grid.point_arrays["values"] = values.flatten().astype(np.float64)  # Flatten the array!
        self.grid.plot(show_edges=True,cmap='jet')

    def plot_field_plt(self, values):
        print('printing')
        nb=self.grid.dimensions
        lb=self.grid.spacing
        sp=self.grid.origin
        # X=np.mgrid[sp[0]:sp[0]+(nb[0]+0.1)*lb[0]:lb[0],
        #               sp[1]:sp[1]+(nb[1]+0.1)*lb[1]:lb[1]]
        X = np.mgrid[sp[0]+0.5*lb[0]:sp[0]+(nb[0]+0.5)*lb[0]:lb[0],
                      sp[1]+0.5*lb[1]:sp[1]+(nb[1]+0.5)*lb[1]:lb[1],
                      sp[2]+0.5*lb[2]:sp[2]+(nb[2]+0.5)*lb[2]:lb[2]]
        # import pdb; pdb.set_trace()
        v=values.reshape(nb[0],nb[1])
        plt.pcolormesh(X[0],X[1],values,cmap='jet')
        ax=plt.gca()
        ax.set_aspect(1)
        plt.show()


    def get_grid(self):
        mesh = inputs['mesh_generation_parameters']
        nb=np.array(mesh['n_blocks'])[[1,0,2]]
        lb=np.array(mesh['block_size'])[[1,0,2]]
        sp=np.array([0,0,0])

        # values=np.arange(nb[0]*nb[1]*nb[2]).reshape(nb)
        grid = pv.UniformGrid()
        grid.dimensions = nb
        grid.origin = sp  # The bottom left corner of the data set
        grid.spacing = lb  # These are the cell sizes along each axis
        return grid
