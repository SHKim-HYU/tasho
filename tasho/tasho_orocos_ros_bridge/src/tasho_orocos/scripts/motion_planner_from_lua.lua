mp = require('motionplanner')

x_0 = {}

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

q0 = {-1.35488912e+00, -8.72846052e-01, 2.18411843e+00,  6.78786296e-01,
  2.08696971e+00, -9.76390128e-01, -1.71721329e+00,  1.65969745e-03,
  1.65969745e-03,  1.47829337e+00, -5.24943547e-01, -1.95134781e+00,
  5.30517837e-01, -2.69960026e+00, -8.14070355e-01,  1.17172289e+00,
  2.06459136e-03,  2.06462524e-03}

a = mp.call_ocp(q,q_dot,s, s_ddot, s_ddot, q_ddot,q0,q_dot0,{0},{0},lam_g)

print(a[199])

