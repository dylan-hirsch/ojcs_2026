# fundamental numerical library
import hj_reachability as hj
import jax.numpy as jnp

# my dynamics
from gra_util.robot import RobotDynamics
from gra_util.warehouse_builder import Warehouse
from gra_util.closed_loop import ClosedLoopTrajectory

def compute_value_downstream(grid, warehouse):

    # specify the running cost
    constraint = warehouse.constraint_sdf(grid)
    target = warehouse.target_sdf(grid)

    def value_postprocessor(t, V):
        return jnp.minimum(jnp.maximum(V, target), constraint)

    # specify the accuracy with which to solve the HJ Partial Differential Equation
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", value_postprocessor=value_postprocessor
    )

    # solve for the value function
    V = hj.solve(
        solver_settings,
        MODEL,
        grid,
        times,
        jnp.minimum(target, constraint),
    )

    return V

if __name__ == "__main__":

    # specify the dynamics parameters
    uA_max = 1.0
    uB_max = 0.5
    d_max = 0.2

    # specify the time horizons of the problem
    B = 60
    S = 40

    # Initial state
    X0 = np.array((-9.0, 9.0))

    # Number of steps to take in closed loop
    STEPS = 5000

    # specify the number of voxels to divide the spatial and temporal axes
    X_RES = 20
    Y_RES = 20
    T_RES = 60

    # Specify PDE grid corners
    X_MIN = 0
    Y_MIN = 0

    X_MIN = 10
    Y_MIN = 10

    # Build the warehouse
    WAREHOUSE = Warehouse()
    warehouse.add_wall(0., 0., 1., 10.)
    warehouse.add_wall(0., 0., 10., 1.)
    warehouse.add_wall(9., 0., 10., 10.)
    warehouse.add_wall(0., 9., 10., 10.)
    warehouse.add_wall(0., 4.5, 4.5, 5.5)
    warehouse.add_wall(5.5, 4.5, 10, 5.5)
    warehouse.add_shelf(4.0, 8.0, 7.0, 9.0)


    # build the dynamics model 
    model = RobotDynamics(uA_max, uB_max ,d_max)

    # build the grid 
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box([X_MIN, Y_MIN], [X_MAX, Y_MAX]),
        [X_RES + 1, Y_RES + 1],
        periodic_dims=None
    )
    times_downstream = np.linspace(B, S, t_voxels + 1)
    times_upstream = np.linspace(S, 0, t_voxels + 1)

    # compute the downstream value function
    #downstream_value = 
    

    # compute the upstream value function


