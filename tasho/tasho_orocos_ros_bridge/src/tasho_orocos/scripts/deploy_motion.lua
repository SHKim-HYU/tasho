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

simulation = true

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
tasho_orocos_dir = rtt.provides("ros"):find("tasho_orocos")
depl:loadComponent("timed_traj", "OCL::LuaComponent")
timed_traj = depl:getPeer("timed_traj")
timed_traj:exec_file(tasho_orocos_dir.."/scripts/ocp_component.lua")
depl:setActivity("timed_traj", 1/samplefreq, 99, rtt.globals.ORO_SCHED_RT)

cp=rtt.Variable("ConnPolicy")

-- deploy simulated robot:
    depl:loadComponent("simrobot", "OCL::LuaComponent")
    simrobot = depl:getPeer("simrobot")
    simrobot:exec_file(tasho_orocos_dir.."/scripts/simple_robot_sim.lua")
    init_jnts=rtt.Variable("array")
    init_jnts:fromtab({-1.35488912e+00, -8.72846052e-01, 2.18411843e+00,  6.78786296e-01,
  2.08696971e+00, -9.76390128e-01, -1.71721329e+00,  1.65969745e-03,
  1.65969745e-03,  1.47829337e+00, -5.24943547e-01, -1.95134781e+00,
  5.30517837e-01, -2.69960026e+00, -8.14070355e-01,  1.17172289e+00,
  2.06459136e-03,  2.06462524e-03})
    simrobot:getProperty("initial_position"):set( init_jnts )
    depl:setActivity("simrobot", 1/samplefreq, 99, rtt.globals.ORO_SCHED_RT)
    timed_traj:getProperty("motion_type"):set(3)

depl:connectPeers("simrobot","timed_traj")
if simulation == true then
    --Run joint_state_publisher.py file in the visualization folder for the visualization
    depl:connect("timed_traj.joint_vel_out_both","simrobot.jointvel",cp ) 
    depl:connect("simrobot.jointpos", "timed_traj.joint_pos_in_both", cp)
    depl:stream("simrobot.jointpos", ros:topic("/joint_states_from_orocos"))
    --depl:stream("timed_traj.joint_pos_ref_out_both", ros:topic("/joint_states_from_orocos"))
    simrobot:configure()
    simrobot:start()

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
    
    timed_traj:getProperty("Tgoal"):set(Tgoal_inp)
    print(Tgoal_inp)
    sleep(0.1)
    timed_traj:configure()    
    sleep(1.1)
    timed_traj:start()

    
    
    sleep(5.0)
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
