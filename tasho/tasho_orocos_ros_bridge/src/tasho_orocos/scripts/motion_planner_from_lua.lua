mp = require('motionplanner')

x_0 = {}

mp.call_ocp(x_0, {})

for i = 1,1062 do
	x_0[i] = 0
end

params = {}
for i = 1,102 do
	params[i] = 0
end
-- a goal pose for the transformation matrix from kinMPC
--np.array([[1.0, 0.0, 0.0, 0.1], [0.0, -1.0, 0.0, -0.4], [0.0, 0.0, -1.0, 0.35], [0, 0, 0, 1]])

--initializing the parameters with a valid transformation matrix
params[36+1] = 1.0;
params[36 + 6] = -1.0
params[36 + 11] = -1.0;
params[36 + 13] = 0.1;
params[36 + 14] = 0.4;
params[36 + 15] = 0.35;
params[36 + 16] = 1.0;
--a = mp.callmp_leftarm(x_0, params)



