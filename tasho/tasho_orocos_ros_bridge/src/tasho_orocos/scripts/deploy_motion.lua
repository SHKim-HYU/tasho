require "rttlib"
require "rttros" 
require "utils"
--require "output_utils_rtt" 

-----------------------------------------------------
use_netcdf = true
samplefreq = 100
-----------------------------------------------------

pp=utils.pp

function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

simulation = false

--rtt.setLogLevel("Debug")
rtt.setLogLevel("Info")

gs=rtt.provides()
tc=rtt.getTC()
if tc:getName() == "lua" then
  depl=tc:getPeer("Deployer")
elseif tc:getName() == "Deployer" then
  depl=tc
end
depl:import("rtt_ros")
ros = gs:provides("ros")
ros:import("etasl_rtt")
ros:import("kdl_typekit")
ros:import("rtt_rospack") --required for rtt.provides("ros"):find(package)
depl:import("rtt_sensor_msgs")
depl:import("rtt_motion_control_msgs")

function s( stringtable )
    local v = rtt.Variable("strings")
    v:fromtab(stringtable)
    return v
end

function d( floattable )
    local v = rtt.Variable("array")
    v:fromtab(floattable)
    return v
end

-- jn = s{"yumi_joint_1_l", "yumi_joint_2_l", "yumi_joint_7_l", "yumi_joint_3_l", "yumi_joint_4_l", "yumi_joint_5_l",
--   "yumi_joint_6_l", "g11", "g12", "yumi_joint_1_r", "yumi_joint_2_r", "yumi_joint_7_r", "yumi_joint_3_r", "yumi_joint_4_r", "yumi_joint_5_r",
--   "yumi_joint_6_r", "g21", "g22"};

--deploy the timed trajectory following componentsimulated robot:
etasl_rtt_dir = rtt.provides("ros"):find("tasho_orocos")
depl:loadComponent("timed_traj", "OCL::LuaComponent")
timed_traj = depl:getPeer("timed_traj")
timed_traj:exec_file(etasl_rtt_dir.."/scripts/ocp_component.lua")
depl:setActivity("timed_traj", 1/samplefreq, 99, rtt.globals.ORO_SCHED_RT)

cp=rtt.Variable("ConnPolicy")

-- deploy simulated robot:
    depl:loadComponent("simrobot", "OCL::LuaComponent")
    simrobot = depl:getPeer("simrobot")
    simrobot:exec_file(etasl_rtt_dir.."/scripts/rtt/simple_robot_sim.lua")
    init_jnts=rtt.Variable("array")
    init_jnts:fromtab({-0.2584968737740488, -1.4751856359575597, 1.8219996281082655, 0.5209283064234688, 1.9839690235976921, 0.26398453977796843, -0.02014585756950192, 0.00014965985730298973, 0.00014965985730298973, 9.238809314721515e-06, -9.316113154953396e-05, 4.120291461621384e-05, -4.938317223991528e-05, 2.4445993726109504e-07, 3.06687470916545e-06, 0.0, 0.00014965985730298973, 0.00014965985730298973})
    simrobot:getProperty("initial_position"):set( init_jnts )
    depl:setActivity("simrobot", 1/samplefreq, 99, rtt.globals.ORO_SCHED_RT)

depl:connectPeers("simrobot","timed_traj")
if simulation == true then
    --Run joint_state_publisher.py file in the visualization folder for the visualization
    depl:connect("timed_traj.joint_vel_out_both","simrobot.jointvel",cp ) 
    depl:connect("simrobot.jointpos", "timed_traj.joint_pos_in_both", cp)
    depl:stream("simrobot.jointpos", ros:topic("/joint_states_from_orocos"))
    --depl:stream("timed_traj.joint_pos_ref_out_both", ros:topic("/joint_states_from_orocos"))
    simrobot:configure()
    simrobot:start()

    --create a sequence of three motions to check the dual arm manipulation planner

    --first left arm
    Tdesired = {}
    Tdesired[1] = 1.0;
    Tdesired[6] = -1.0
    Tdesired[11] = -1.0;
    Tdesired[13] = 0.4;
    Tdesired[14] = 0.2;
    Tdesired[15] = 0.65;
    Tdesired[16] = 1.0;
    local Tgoal_inp = rtt.Variable("array")
    Tgoal_inp:fromtab(Tdesired)
    timed_traj:getProperty("gain"):set(0)
    timed_traj:getProperty("dt"):set(1/samplefreq)
    timed_traj:getProperty("motion_type"):set(1.0)
    timed_traj:getProperty("Tgoal"):set(Tgoal_inp)
    print(Tgoal_inp)
    sleep(0.1)
    timed_traj:configure()    
    sleep(5.1)
    timed_traj:start()

    
    
    sleep(9.0)

    timed_traj:stop()

    --second right arm
    Tdesired = {}
    Tdesired[1] = 1.0;
    Tdesired[6] = -1.0
    Tdesired[11] = -1.0;
    Tdesired[13] = 0.4;
    Tdesired[14] = -0.4;
    Tdesired[15] = 0.65;
    Tdesired[16] = 1.0;
    local Tgoal_inp = rtt.Variable("array")
    Tgoal_inp:fromtab(Tdesired)
    timed_traj:getProperty("motion_type"):set(2.0)
    timed_traj:getProperty("Tgoal"):set(Tgoal_inp)
    print(Tgoal_inp)
    sleep(0.1)
    timed_traj:configure()    
    sleep(5.1)
    timed_traj:start()

    
    
    sleep(10.0)

    timed_traj:stop()

    --now dual arm

    Tdesired = {}
    Tdesired[1] = 1.0;
    Tdesired[6] = -1.0
    Tdesired[11] = -1.0;
    Tdesired[13] = 0.4;
    Tdesired[14] = -0.2;
    Tdesired[15] = 0.45;
    Tdesired[16] = 1.0;
    local Tgoal_inp = rtt.Variable("array")
    Tgoal_inp:fromtab(Tdesired)
    timed_traj:getProperty("motion_type"):set(3.0)
    timed_traj:getProperty("Tgoal"):set(Tgoal_inp)
    print(Tgoal_inp)
    sleep(0.1)
    timed_traj:configure()    
    sleep(5.1)
    timed_traj:start()

    
    
    sleep(10.0)
    timed_traj:stop()
    simrobot:stop()
else 
    depl:import("abb_egm_driver")
        depl:loadComponent("yumi", "EGM::EGMDriver")
        yumi = depl:getPeer("yumi")

        --Configuration
        --6511 is ROB_L 6512 is ROB_R
        yumi:getProperty("simulation"):set(false)
        yumi:getProperty("egm_ip"):set("192.168.125.1")
        yumi:getProperty("egm_port"):set(6512)
        yumi:getProperty("egm_rate"):set(100)
        depl:setActivity("yumi", 0, 99, rtt.globals.ORO_SCHED_RT)

        depl:loadComponent("yumi_l", "EGM::EGMDriver")
        yumi_l = depl:getPeer("yumi_l")

        --Configuration
        --6511 is ROB_L 6512 is ROB_R
        yumi_l:getProperty("simulation"):set(false)
        yumi_l:getProperty("egm_ip"):set("192.168.125.1")
        yumi_l:getProperty("egm_port"):set(6511)
        yumi_l:getProperty("egm_rate"):set(100)
        depl:setActivity("yumi_l", 0, 99, rtt.globals.ORO_SCHED_RT)

        dir = rtt.provides("ros"):find("abb_egm_driver") .. "/scripts/"
        
        depl:connect("yumi.JointVelocityCommand", "timed_traj.joint_vel_out_r", cp)
        depl:connect("timed_traj.joint_pos_in_r", "yumi.q_actual", cp)
        depl:connect("yumi_l.JointVelocityCommand", "timed_traj.joint_vel_out_l", cp)
        depl:connect("timed_traj.joint_pos_in_l", "yumi_l.q_actual", cp)
        depl:stream("timed_traj.joint_pos_ref_out_both", ros:topic("/joint_states_from_orocos"))

        -- configure and start-up
        yumi:configure()
        yumi_l:configure()
    
        yumi:start()
        yumi_l:start()
        sleep(0.1)
        Tdesired = {}

        left_arm = false
        right_arm = true
        timed_traj:getProperty("gain"):set(0)
        timed_traj:getProperty("simulation"):set(false)

        if left_arm then
            --left arm
            Tdesired[1] = 1.0;
            Tdesired[6] = -1.0
   
            Tdesired[11] = -1.0;
            Tdesired[13] = 0.45;
            Tdesired[14] = 0.2;
            Tdesired[15] = 0.45;
            Tdesired[16] = 1.0;
            local Tgoal_inp = rtt.Variable("array")
            Tgoal_inp:fromtab(Tdesired)
            timed_traj:getProperty("motion_type"):set(1.0)
            timed_traj:getProperty("Tgoal"):set(Tgoal_inp)

        elseif right_arm then
            timed_traj:getProperty("gain"):set(0)
            Tdesired[1] = -1.0;
            Tdesired[6] = 1.0
    
            Tdesired[11] = -1.0;
            Tdesired[13] = 0.45;
            Tdesired[14] = -0.2;
            Tdesired[15] = 0.45;
            Tdesired[16] = 1.0;
            local Tgoal_inp = rtt.Variable("array")
            Tgoal_inp:fromtab(Tdesired)
            timed_traj:getProperty("motion_type"):set(3.0)
            timed_traj:getProperty("Tgoal"):set(Tgoal_inp)

        end

        timed_traj:configure()    
        sleep(5.01)
        timed_traj:start()

        --reporter:start();
    
           
    print("configuration of the orocos components done")


    sleep(10.00)

    yumi:stop()
    yumi_l:stop()
    sleep(0.01)
    timed_traj:stop()

end
