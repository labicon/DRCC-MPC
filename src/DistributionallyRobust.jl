#///////////////////////////////////////
#// File Name: DistributionallyRobust.jl
#// Author: Kanghyun Ryu (kr37@illinois.edu)
#// Date Created: 2023/03/30
#// Description: Julia package for Distributionally Robust 
#// Control, with public version of Trajectron++
#///////////////////////////////////////

module DistributionallyRobust

import Base: ==, isequal, isapprox
import PlotUtils: plot_color
import RobotOS: Time, Duration
import StatsBase: fit, Histogram
import StatsFuns: logsumexp

using CUDA
using DataStructures
using Distributions
using ForwardDiff
using LinearAlgebra
using Plots
pyplot();
using Printf
using ProgressMeter
using PyCall
using Random

# # Modules for Buffered Input Cell and Related Functions (and constrained SAC optimizations)
# import Convex: Variable, norm, quadform, minimize, dot, solve!
# using GeometricalPredicates
# using LazySets
# using Polyhedra
# using SCS
# using ECOS
# using VoronoiCells

function __init__()
    @info "Number of Julia Thread(s): $(Threads.nthreads())";
    @info "CUDA Device: $(name(CuDevice(0)))"
    @info "Python executable used by PyCall: $(pyimport("sys").executable)";

    # Append the code directory to python path
    # (see https://github.com/JuliaPy/PyCall.jl#troubleshooting)
    file_path = @__FILE__;
    pushfirst!(PyVector(pyimport("sys")."path"),
               py"'/'.join($file_path.split('/')[:-1]) + '/../Trajectron-plus-plus/trajectron'");
    pushfirst!(PyVector(pyimport("sys")."path"),
               py"'/'.join($file_path.split('/')[:-1]) + '/../CrowdNav'");

    # Define python functions for Trajectron bridge
    py"""
    import os
    import time
    import json
    import torch
    import dill
    import random
    import pathlib
    import pandas as pd
    # import evaluation
    import numpy as np
    import visualization as vis
    from easydict import EasyDict
    from model.online.online_trajectron import OnlineTrajectron
    from model.model_registrar import ModelRegistrar
    from environment import Environment, Scene, Node

    # For CrowdNav Baseline
    import configparser
    import crowd_nav
    import gym
    from crowd_nav.policy.policy_factory import policy_factory
    from crowd_sim.envs.utils.robot import Robot
    from crowd_sim.envs.utils.state import ObservableState

    def load_hyperparams(model_dir, conf_file):
        '''
        param model_dir: directory where the learned model resides
        param conf_file: config file name
        '''
        # Load hyperparameters from json
        config_file = os.path.join(model_dir, conf_file)
        if not os.path.exists(config_file):
            raise ValueError('Config json not found!')
        with open(config_file, 'r') as conf_json:
            hyperparams = json.load(conf_json)
        return hyperparams

    def load_online_scene(data_dir, data_name, maximum_history_length, state_def,
                         scene_idx, incl_robot_node, init_timestep=2,
                         ado_id_removed=None):

        '''
        param data_dir: directory to look in for evaluation data
        param data_name: name of the evaluation data
        param maximum_history_length: maximum history length considered in the model
        param state_def: state definition
        param scene_idx: scene_idx (starting from 0)
        param incl_robot_node: whether to include the robot node in the scene
        param init_timestep: init_timestep (starting form 0). You need to have at least acceleration, so you want 2 timesteps of prior data, e.g. [0, 1],
                             so that you can immediately start incremental inference from the 3rd timestep onwards.
        param ado_id_removed: ado_id in string to be removed from online_scene. Note this ado will not be removed from eval_scene
        '''
        log = []
        eval_data_path = os.path.join(data_dir, data_name)
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')
        log.append('Loaded evaluation data from %s' % (eval_data_path,))
        if eval_env.robot_type is None and incl_robot_node:
             eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?

             x = [np.nan]
             y = [np.nan]
             vx = [np.nan]
             vy = [np.nan]
             ax = [np.nan]
             ay = [np.nan]
             data_dict = {('position', 'x'): x,
                          ('position', 'y'): y,
                          ('velocity', 'x'): vx,
                          ('velocity', 'y'): vy,
                          ('acceleration', 'x'): ax,
                          ('acceleration', 'y'): ay}
             data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
             node_data = pd.DataFrame(data_dict, columns=data_columns)
             robot_node = Node(eval_env.robot_type, 'ROBOT', node_data, is_robot=True, first_timestep=init_timestep)
             for scene in eval_env.scenes:
                 #scene.add_robot_from_nodes(eval_env.robot_type)
                 scene.robot = robot_node
                 scene.nodes.append(scene.robot)

        assert 0 <= scene_idx < len(eval_env.scenes), "scene_idx must be between 0 and {}!".format(len(eval_env.scenes) - 1)
        eval_scene = eval_env.scenes[scene_idx]

        online_scene = Scene(timesteps=init_timestep,
                             map=eval_scene.map,
                             dt=eval_scene.dt)
        online_scene.nodes = eval_scene.get_nodes_clipped_at_time(timesteps=np.arange(init_timestep + 1 - maximum_history_length,
                                                                                      init_timestep),
                                                                  state=state_def)
        online_scene.robot = eval_scene.robot
        node_to_remove = None
        if ado_id_removed is not None:
            for node in online_scene.nodes:
                if str(node) == ado_id_removed:
                    node_to_remove = node
        if node_to_remove is not None:
            online_scene.nodes.remove(node_to_remove)

        return eval_env, eval_scene, online_scene, log

    def initialize_model(args, model_dir, conf_file):
        '''
        param args: easydict object for parsing arguments
        param model_dir: directory where the learned model resides
        param confi_file: config file name
        '''
        log = []
        # Load hyperparameters from json
        hyperparams = load_hyperparams(model_dir, conf_file)

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        # Add hyperparams from arguments
        hyperparams['dynamic_edges'] = args.dynamic_edges
        hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
        hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
        hyperparams['edge_addition_filter'] = args.edge_addition_filter
        hyperparams['edge_removal_filter'] = args.edge_removal_filter
        hyperparams['offline_scene_graph'] = args.offline_scene_graph
        hyperparams['incl_robot_node'] = args.incl_robot_node
        hyperparams['edge_encoding'] = not args.no_edge_encoding
        hyperparams['use_map_encoding'] = args.map_encoding

        iter_num = 100
        model_registrar = ModelRegistrar(model_dir, args.eval_device)
        model_registrar.load_models(iter_num=iter_num)
        log.append('Loaded Trajectron model from %s' % (os.path.join(model_dir, 'model_registrar-{}.pt'.format(iter_num))))
        trajectron = OnlineTrajectron(model_registrar, hyperparams,
                                  args.eval_device)
        return trajectron, hyperparams, log

    def configure_rl_robot(model_dir, env_config_path, policy_config_path,
                           policy_name='sarl'):
        env_config_file = os.path.join(model_dir, os.path.basename(env_config_path))
        policy_config_file = os.path.join(model_dir, os.path.basename(policy_config_path))
        model_weights = os.path.join(model_dir, 'rl_model.pth')
        device = torch.device('cpu')

        # configure policy
        policy = policy_factory[policy_name]()
        policy_config = configparser.RawConfigParser()
        policy_config.read(policy_config_file)
        policy.configure(policy_config)
        if policy.trainable:
            policy.get_model().load_state_dict(torch.load(model_weights))
        phase = 'test'
        policy.set_phase(phase)
        policy.set_device(device)
        if hasattr(policy, 'query_env'):
            policy.query_env = False

        # configure environment
        env_config = configparser.RawConfigParser()
        env_config.read(env_config_file)

        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)

        # additional setup (from crowd_sim.CrowdSim.reset)
        time_step = env_config.getfloat('env', 'time_step')
        robot.time_step = time_step
        robot.policy.time_step = time_step
        if robot.sensor == 'RGB':
            raise NotImplementedError

        return robot
    """
end

# Type Definition
export
    Parameter,
    State,
    WayPoint2D,
    Trajectory2D,
    get_position,
    RobotState,
    get_velocity,
    WorldState
include("type_definition.jl")

# State Transition
export
    transition,
    transition_jacobian,
    transition_control_coeff
    #sample_future_positions
include("state_transition.jl")

# Cost
export
    instant_position_cost,
    instant_control_cost,
    instant_collision_cost,
    terminal_position_cost,
    terminal_collision_cost,
    DRCCostParameter
include("drc_cost.jl")

# Cost (CUDA kernels)
export
    kernel_instant_position_cost!,
    kernel_instant_control_cost!,
    kernel_instant_collision_cost!,
    kernel_terminal_position_cost!,
    kernel_terminal_collision_cost!
include("drc_cost_gpu.jl")

# Scene Loader
export
    fetch_ado_positions!,
    convert_nodes_to_str,
    SceneLoader,
    TrajectronSceneParameter,
    TrajectronSceneLoader,
    reduce_to_positions,
    get_trajectory_for_ado,
    SyntheticSceneParameter,
    SyntheticSceneLoader
include("scene_loader.jl")

# Predictor
export
    initialize_scene_graph!,
    sample_future_ado_positions!,
    Predictor,
    TrajectronPredictorParameter,
    TrajectronPredictor,
    OraclePredictorParameter,
    OraclePredictor,
    GaussianPredictorParameter,
    GaussianPredictor
include("predictor.jl")

# Forward-Backward Simulation
export
    get_measurement_schedule,
    get_target_pos_array,
    process_u_arrays,
    simulate_forward,
    sample_ado_positions,
    process_ap_dict,
    compute_costs,
    integrate_costs,
    choose_best_nominal_control,
    compute_cost_gradients,
    sum_cost_gradients,
    simulate_backward,
    simulate,
    # evaluate_risk,
    SimulationParameter,
    SimulationCostResult,
    SimulationCostGradResult,
    SimulationResult
include("drc_forward_backward_simulation.jl")

# Distributionally Robust Controller
export
    DRCController,
    control!,
    adjust_old_prediction!,
    schedule_prediction!,
    schedule_control_update!,
    drc_control_update!,
    get_action!,
    cem_optimization!,
    get_mean_cov,
    compute_cost_CvaR,
    compute_cost,
    compute_CVaR,
    compute_CVaR_array,
    compute_CVaR_array_gpu,
    kernel_CVaR!,
    get_robot_present_and_future,
    DRCControlParameter
include("distributionally_robust_controller.jl")

# Evaluation Functions
export
    evaluate,
    EvaluationResult,
    BICEvaluationResult
include("drc_evaluation.jl")

# Helper Functions
export
    get_nominal_trajectory,
    init_condition_setup,
    controller_setup,
    display_log,
    visualize!,
    make_gif,
    fetch_stats_filtered,
    plot_histogram
include("drc_utils.jl")

end # module