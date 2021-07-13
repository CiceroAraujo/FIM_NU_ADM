from ..preprocessor.multiscale import get_primal_and_dual_meshes
class NewtonIterationMultilevel:
    def __init__(self, wells, faces, volumes):
        self.GID_1, self.DUAL_1 = get_primal_and_dual_meshes(volumes['centroids'], faces)


    def multiscale_preprocessor(self):
        import pdb; pdb.set_trace()
