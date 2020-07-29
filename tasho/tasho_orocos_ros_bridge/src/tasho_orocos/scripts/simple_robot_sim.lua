require("rttlib")
require("math")

tc=rtt.getTC()

iface_spec = {
   ports={
      { name='jointpos', datatype='array', type='out', desc="joint positions corresponding to the integrated joint velocities" },
      { name='jointvel', datatype='array', type='in', desc="joint velocities" },
   },
 
   properties={
      { name='initial_position', datatype='array', desc="initial position of the joints" }
   }
}


iface=rttlib.create_if(iface_spec)

jvals=rtt.Variable("array")

-- The Lua component starts its life in PreOperational, so
-- configureHook can be used to set stuff up.
function configureHook()
    iface=rttlib.create_if(iface_spec)
    local vals=iface.props.initial_position:get():totab()
    jvals:fromtab( vals )
    iface.ports.jointpos:write(jvals) 
    return true
end

function startHook()
    local vals=iface.props.initial_position:get():totab()
    jvals:fromtab( vals )
    iface.ports.jointpos:write(jvals) 
    return true
end


function updateHook()
    --print("simple_robot_sim::updateHook")
    fs,vel=iface.ports.jointvel:read()
    if fs~='NoData' then 
        local v = vel:totab()
        local dt=tc:getPeriod()        
        for i=1,#v do
            -- the arrays are zero-based index, while in LUA it is one-based index.
            -- Markus: you made life difficult and confusing !
            jvals[i-1] = jvals[i-1] + v[i] * dt
        end
    end
    iface.ports.jointpos:write(jvals) 
end


function cleanupHook()
    rttlib.tc_cleanup()
end
