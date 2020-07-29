-- A component to follow the timed trajectory of the ocp output is followed
-- Suitable for cases like freespace motion where not much plant-model mismatch is expected
-- ATTENTION: It is assumed that the robot starts from a rest state.

-- Given a sequence of joint position and velocity values, interpolate to 
-- to provide a piecewise constant acceleration trajectory to the robot

require("rttlib")
require("math")
require("context")
require("geometric")
mp = require("motionplanner")
require "utils"
require "rttros" 

tc=rtt.getTC()
-- the depl will be nil, since for some reason it is not a peer of our component, even if stated explicitly in the deploy script
if tc:getName() == "lua" then
  depl=tc:getPeer("Deployer")	
elseif tc:getName() == "Deployer" then
  depl=tc
end

iface_spec = {
   ports={
      { name='joint_vel_out_r', datatype='motion_control_msgs/JointVelocities', type='out', desc="joint velocity right arm output" },
      { name='joint_vel_out_l', datatype='motion_control_msgs/JointVelocities', type='out', desc="joint velocity left arm output" },
      { name='joint_vel_out_both', datatype='array', type='out', desc="joint velocity both arm output" },
      { name='joint_pos_ref_out_both', datatype='array', type='out', desc="joint position both arm output" },
      { name='joint_pos_in_both', datatype='array', type='in', desc="joint position both arm output" },
      { name='joint_pos_in_l', datatype='array', type='in', desc="joint position left arm inp" },
      { name='joint_pos_in_r', datatype='array', type='in', desc="joint position right arm inp" },
   },
 
   properties={
      { name='motion_type', datatype='double', desc="Type of motion. 1- left arm, 2- right arm, 3-cooperative manipulation" },
      { name='Tgoal', datatype='array', desc="rolled vector of the transformation matrix" },
      { name='Ts', datatype='double', desc="Sampling time of the trajectory" },
      { name='gain', datatype='double', desc="proportional feedback gain on the joint position" },
      { name='ndof', datatype='int', desc="number of degrees of freedom of the system" },
      { name='no_samples', datatype='int', desc="number of trajectory points sent to the system" },
      { name='joint_pos_traj', datatype='array', desc="joint position trajectory input" },
      { name='joint_vel_traj', datatype='array', desc="joint velocity trajectory input" },
      { name='dt', datatype='double', desc="sample time of the controller" },
      { name='simulation', datatype='bool', desc="whether run in simulation mode" },
   }
}

--TODO: convert Tgoal to KDL frame datatype
counter = 0
iface=rttlib.create_if(iface_spec)
iface.props.Ts:set(0.4)
iface.props.ndof:set(18)
iface.props.no_samples:set(11)
iface.props.gain:set(0)
iface.props.simulation:set(true)
iface.props.dt:set(1/250.0)
--jvals:fromtab({1.0, 0.0, 1.1, 0.0})
--jvvals = rtt.Variable("array")
--jvvals:fromtab({0.0, 0.0, 1.0, 0.0})
--iface.props.joint_pos_traj:set(jvals)
--iface.props.joint_vel_traj:set(jvvals)

time_start = 0
motion_start = false
joint_pos_ref_out = rtt.Variable("array")
joint_vel_ref_out = rtt.Variable("array")
joint_vel_ref_l_out =rtt.Variable("motion_control_msgs/JointVelocities")
joint_vel_ref_r_out =rtt.Variable("motion_control_msgs/JointVelocities")
joint_pos_ref_out:fromtab({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
joint_vel_ref_out:fromtab({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
joint_pos_val = {0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0}
joint_vel_ref_r = {0, 0, 0, 0, 0, 0, 0}
joint_vel_ref_l = {0, 0, 0, 0, 0, 0, 0}

  
joint_vel_traj = {} --iface.props.joint_vel_traj:get()
joint_pos_traj = {} --iface.props.joint_pos_traj:get() 

function configureHook()

  gain= iface.props.gain:get()
  Ts= iface.props.Ts:get()
  ndof= iface.props.ndof:get()
  no_samples= iface.props.no_samples:get()
  dt = iface.props.dt:get()
  simulation = iface.props.simulation:get()
  motion_type =  iface.props.motion_type:get()

  motion_start = false
  time_start = 10000 -- Just initializing with a large value
  time_end = no_samples*Ts -- The total time duration of the timed trajectory
  --should read the robot joint pose here, then send it to the casadi solver as one of the opti.x and also the goal frame should be passed.
  iface=rttlib.create_if(iface_spec)

  q = {}
  for i=1,198 do 
  	q[i] = 0
  end
  q_dot = {}
  for i=1,198 do 
  	q_dot[i] = 0
  end
  s = {}
  for i=1,11 do 
  	s[i] = 0
  end
  s_dot = {}
  for i=1,11 do 
  	s_dot[i] = 0
  end
  s_ddot = {}
  for i=1,10 do 
  	s_ddot[i] = 0
  end
  q_ddot = {}
  for i=1,180 do 
  	q_ddot[i] = 0
  end
  
  lam_g = {}
  for i=1,1108 do 
  	lam_g[i] = 0
  end
  
  q_dot0 = {}
  for i=1,18 do 
  	q_dot0[i] = 0
  end
  
  s0 = {}
  s0[1] = 0
  
  s_dot0 = {}
  s_dot0[1] = 0
  
  -- q0 = {-1.35488912e+00, -8.72846052e-01, 2.18411843e+00,  6.78786296e-01,
  --   2.08696971e+00, -9.76390128e-01, -1.71721329e+00,  1.65969745e-03,
  --   1.65969745e-03,  1.47829337e+00, -5.24943547e-01, -1.95134781e+00,
  --   5.30517837e-01, -2.69960026e+00, -8.14070355e-01,  1.17172289e+00,
  --   2.06459136e-03,  2.06462524e-03}

  q0 = {}


  fs,pos=iface.ports.joint_pos_in_both:read()
  if fs~='NoData' then 
        local p = pos:totab()
        for i = 1, 18 do
        	q0[i] = p[i]
        end
    end


   fs,pos=iface.ports.joint_pos_in_l:read()
  if fs~='NoData' then 
        local p = pos:totab()
        for i = 1, 9 do
        	--print("i is " .. i)
        	q0[i] = p[i]
        	--print(type(joint_pos_val[i]))
        end
    end

    fs,pos=iface.ports.joint_pos_in_r:read()
  if fs~='NoData' then 
        local p = pos:totab()
        for i = 1, 9 do
        	--print("i is " .. i)
        	q0[i+9] = p[i]
        	--print(type(joint_pos_val[i]))
        end
    end






-- a goal pose for the transformation matrix from kinMPC
--np.array([[1.0, 0.0, 0.0, 0.1], [0.0, -1.0, 0.0, -0.4], [0.0, 0.0, -1.0, 0.35], [0, 0, 0, 1]])

--initializing the parameters with a valid transformation matrix
-- params[36+1] = 1.0;
-- params[36 + 6] = -1.0
-- params[36 + 11] = -1.0;
-- params[36 + 13] = 0.4;
-- params[36 + 14] = 0.4;
-- params[36 + 15] = 0.65;
-- params[36 + 16] = 1.0;
-- local Tgoal= iface.props.Tgoal:get()
-- local 
-- for i = 1,16 do
--   params[36 + i] = Tgoal[i-1]
-- end
if motion_type == 1.0 then

  motion_plan = mp.callmp_leftarm(x_0, params)

elseif motion_type == 2.0 then

  motion_plan = mp.callmp_rightarm(x_0, params)

elseif motion_type == 3.0 then
  print("Calling the casadi OCP solver from lua")
  motion_plan = mp.call_ocp(q,q_dot,s, s_ddot, s_ddot, q_ddot,q0,q_dot0,{0},{0},lam_g)

else 

  print("NO valid motion type specified. The specified motion type was")
  print(Tgoal)

end

a = {}
b = {}
  for i = 1,18*no_samples do
  	b[i] = motion_plan[i]
  	a[i] = motion_plan[18*no_samples+i]
  end
  jvvals = rtt.Variable("array")
  jvvals:fromtab(a)
  jvals=rtt.Variable("array")
  jvals:fromtab(b)
  iface.props.joint_pos_traj:set(jvals)
  iface.props.joint_vel_traj:set(jvvals)

  joint_vel_traj = iface.props.joint_vel_traj:get()
  joint_pos_traj = iface.props.joint_pos_traj:get()
  gain= iface.props.gain:get()
  Ts= iface.props.Ts:get()
  ndof= iface.props.ndof:get()
  no_samples= iface.props.no_samples:get()
  dt = iface.props.dt:get()
  simulation = iface.props.simulation:get() 
  -- print("jval is ")
  -- print(jvvals)
  --jvvals:fromtab(motion_plan[1,1062])
  return true
end

function startHook()
  local a,b = rtt.getTime()
  time_start = a+b*1e-9
  return true
end

function updateHook()
   local a,b = rtt.getTime()
    local time = a+b*1e-9      

    if motion_start then
      time = time - time_start

      -- if time <= 5 then
      --   counter = counter + 1 --to check update hook rate when we do practically nothing
      --   return
      -- end
    else
      print("starting joint motion in updateHook")
      
      time_start = time
      time = 0
      motion_start = true
    end

  if simulation then
   fs,pos=iface.ports.joint_pos_in_both:read()
   if fs~='NoData' then 
        local p = pos:totab()
        for i = 1, 18 do
        	--print("i is " .. i)
        	joint_pos_val[i] = p[i]
        	--print(type(joint_pos_val[i]))
        end
    end

  else  

    fs,pos=iface.ports.joint_pos_in_l:read()
    if fs~='NoData' then 
        local p = pos:totab()
        for i = 1, 7 do
        	--print("i is " .. i)
        	joint_pos_val[i] = p[i]
        	--print(type(joint_pos_val[i]))
        end
    end

    fs,pos=iface.ports.joint_pos_in_r:read()
    if fs~='NoData' then 
        local p = pos:totab()
        for i = 1, 7 do
        	--print("i is " .. i)
        	joint_pos_val[i+9] = p[i]
        	--print(type(joint_pos_val[i]))
        end
    end

  end

  --print(joint_pos_val)

    
   
    
    --joint_vel_ref_r:fromtab({0, 0, 0, 0, 0, 0, 0, 0, 0})
    --joint_vel_ref_l:fromtab({0, 0, 0, 0, 0, 0, 0, 0, 0})
    --print("time is ".. time)
    --print("first joint pose " .. joint_pos_traj[1])
    --print("first velocity pose " .. joint_vel_traj[1])

    -- print(type(joint_pos_traj))
    -- print("no_samples " .. no_samples)
    --print("joint pos val is ")
    
    if time <= time_end then
      counter = counter + 1;
    	vel_index = math.floor(time / Ts)
    	--print("vel index is" .. vel_index)
    	for i = 0, ndof-1 do
    		acc_val = (joint_vel_traj[(vel_index + 1)*ndof + i] - joint_vel_traj[vel_index*ndof + i]) / Ts
    		joint_pos_ref_out[i] = joint_pos_traj[vel_index*ndof + i] + (time - Ts*(vel_index))*joint_vel_traj[vel_index*ndof + i] + 0.5*(time - Ts*(vel_index))^2*acc_val
    		joint_vel_ref_out[i] = joint_vel_traj[vel_index*ndof + i] + (time + dt/2 - Ts*(vel_index))*acc_val 
    		--joint_vel_ref_out[i] = joint_vel_ref_out[i] + gain*(joint_pos_ref_out[i] - joint_pos_val[i+1])  
    		-- print("i is " .. i)
    		-- print("acc_val is " .. acc_val)
    		-- print("joint pos ref is " .. joint_pos_ref_out[i])
    		-- print("joint vel ref is " .. joint_vel_ref_out[i])
		end
    --print(joint_vel_ref_out)
		   else
        --print("Time out")
        for i = 0, ndof-1 do
        joint_vel_ref_out[i] = 0
        joint_vel_ref_out[i] = joint_vel_ref_out[i] + 0*(joint_pos_ref_out[i] - joint_pos_val[i+1])  
        -- print("i is " .. i)
        -- print("acc_val is " .. acc_val)
        -- print("joint pos ref is " .. joint_pos_ref_out[i])
        -- print("joint vel ref is " .. joint_vel_ref_out[i])
    end
       -- joint_vel_ref_out:fromtab({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
    	-- send an event that the trajectory is finished
    	--return
    end

    if not simulation then

		  for i = 0,6 do 
			 joint_vel_ref_l[i+1] = joint_vel_ref_out[i]
			 joint_vel_ref_r[i+1] = joint_vel_ref_out[i+9]
		  end

    end
		
    iface.ports.joint_pos_ref_out_both:write(joint_pos_ref_out)
    iface.ports.joint_vel_out_both:write(joint_vel_ref_out)
    joint_vel_ref_l_out:fromtab({names={}, velocities=joint_vel_ref_l})
    iface.ports.joint_vel_out_l:write(joint_vel_ref_l_out)
    joint_vel_ref_r_out:fromtab({names={}, velocities=joint_vel_ref_r})
    iface.ports.joint_vel_out_r:write(joint_vel_ref_r_out)

    --print(joint_pos_ref_out)
    --print(joint_vel_ref_out)
    --print(joint_vel_ref_l_out)
    --print(joint_vel_ref_r_out)
    --print("\n\n")

 

end

function stopHook()
  print(joint_pos_ref_out)
	print("Timed trajectory completed")
  print("counter value = ".. counter)

end	
