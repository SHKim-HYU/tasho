#Code to analyze the data of the benchmarking test that is stored in the json form
import json
import matplotlib.pyplot as plt
# import tkinter
from pylab import *

#Load the json data
with open('examples/wip/tasho_paper_examples/p2p_L1_terminal_cost.txt', 'r') as fp:
	results = json.load(fp)

#begin averaging the data
horizon_sizes = [2,3,4,5,6,7,8,9,10,12,14,16,18,20, 24, 28, 32, 36, 40, 50]

#Creating an empty array for storing all parameters for different horizon sizes
feasibility_total = {}
average_solver_time_total = {}
motion_time_total = {}
torque_effort_total = {}

for horizon_size in horizon_sizes:

	feasibility_total[horizon_size] = []
	average_solver_time_total[horizon_size] = []
	motion_time_total[horizon_size] = []
	torque_effort_total[horizon_size] = []


#begin aggregating the data 
for key in results:

	key_split = key.split(',')
	h_size = int(key_split[0])
	eg_no = int(key_split[1])

	if results[key]['status'] == 'MPC_SUCCEEDED':

		feasibility_total[h_size].append(eg_no)
		average_solver_time_total[h_size].append(results[key]['avg_solver_time'])
		motion_time_total[h_size].append(results[key]['total_trajectory_time'])
		torque_effort_total[h_size].append(results[key]['torque_effort'][0])


#computing the averages
feasibility_average = {}
infeasibilty_percentage = []
average_solver_time_average = {}
motion_time_average = {}
torque_effort_average = {}

for horizon_size in horizon_sizes:

	feasibility_average[horizon_size] = len(feasibility_total[horizon_size])/50
	infeasibilty_percentage.append((1 - feasibility_average[horizon_size])*100)
	average_solver_time_average[horizon_size] = sum(average_solver_time_total[horizon_size])/len(average_solver_time_total[horizon_size])
	motion_time_average[horizon_size] = sum(motion_time_total[horizon_size])/len(motion_time_total[horizon_size])
	torque_effort_average[horizon_size] = sum(torque_effort_total[horizon_size])/len(torque_effort_total[horizon_size])

print(feasibility_average)
print(infeasibilty_percentage)
print(average_solver_time_average)
print(motion_time_average)
print(torque_effort_average)

#Visualization for L2 cost

with open('examples/wip/tasho_paper_examples/p2p_L2_terminal_cost.txt', 'r') as fp:
	results = json.load(fp)

#begin averaging the data
horizon_sizes = [2,3,4,5,6,7,8,9,10,12,14,16,18,20, 24, 28, 32, 36, 40, 50]

#Creating an empty array for storing all parameters for different horizon sizes
feasibility_total = {}
average_solver_time_total = {}
motion_time_total = {}
torque_effort_total = {}

for horizon_size in horizon_sizes:

	feasibility_total[horizon_size] = []
	average_solver_time_total[horizon_size] = []
	motion_time_total[horizon_size] = []
	torque_effort_total[horizon_size] = []


#begin aggregating the data 
for key in results:

	key_split = key.split(',')
	h_size = int(key_split[0])
	eg_no = int(key_split[1])

	if results[key]['status'] == 'MPC_SUCCEEDED':

		feasibility_total[h_size].append(eg_no)
		average_solver_time_total[h_size].append(results[key]['avg_solver_time'])
		motion_time_total[h_size].append(results[key]['total_trajectory_time'])
		torque_effort_total[h_size].append(results[key]['torque_effort'][0])


#computing the averages
feasibility_average_L2 = {}
infeasibilty_percentage_L2 = []
average_solver_time_average_L2 = {}
motion_time_average_L2 = {}
torque_effort_average_L2 = {}

for horizon_size in horizon_sizes:

	feasibility_average_L2[horizon_size] = len(feasibility_total[horizon_size])/50
	infeasibilty_percentage_L2.append((1 - feasibility_average_L2[horizon_size])*100)
	average_solver_time_average_L2[horizon_size] = sum(average_solver_time_total[horizon_size])/len(average_solver_time_total[horizon_size])
	motion_time_average_L2[horizon_size] = sum(motion_time_total[horizon_size])/len(motion_time_total[horizon_size])
	torque_effort_average_L2[horizon_size] = sum(torque_effort_total[horizon_size])/len(torque_effort_total[horizon_size])

print(feasibility_average_L2)
print(infeasibilty_percentage_L2)
print(average_solver_time_average_L2)
print(motion_time_average_L2)
print(torque_effort_average_L2)

figure()
horizon_sizes_left = []
horizon_sizes_right = []
for h in horizon_sizes:
	horizon_sizes_left.append(h - 0.2)
	horizon_sizes_right.append(h + 0.2)
bar(horizon_sizes_left, infeasibilty_percentage, width = 0.4, label = 'L1 penalty')
bar(horizon_sizes_right, infeasibilty_percentage_L2,  width = 0.4, label = 'L2 penalty')
legend()
title("Infeasibility of P2P motion vs Horizon size")
xlabel("No of samples in the horizon (sampling time = 0.05s)")
ylabel("Percentage of Infeasibility (%)")
show()

figure()
plot(horizon_sizes, list(average_solver_time_average.values()), label = 'L1 penalty')
plot(horizon_sizes, list(average_solver_time_average_L2.values()), label = 'L2 penalty')
legend()
title("Average solver time per MPC iteration vs Horizon size")
xlabel("No of samples in the horizon (sampling time = 0.05s)")
ylabel('Seconds')
show(block=True)

figure()
plot(horizon_sizes, list(motion_time_average.values()), label = 'L1 penalty')
plot(horizon_sizes, list(motion_time_average_L2.values()), label = 'L2 penalty')
legend()
axis([0, 60, 0, 13])
title("Average P2P motion time vs Horizon size")
xlabel("No of samples in the horizon (sampling time = 0.05s)")
ylabel('Seconds')
show(block=True)

figure()
plot(horizon_sizes, list(torque_effort_average.values()), 'bx', label = 'L1 penalty')
plot(horizon_sizes, list(torque_effort_average_L2.values()), 'rx', label = 'L2 penalty')
legend()
title("Average sum of square torque_effort integrated over time vs Horizon size")
ylabel('N^2m^2s')
show(block=True)



#Find tasks not feasible for any horizon
task_examples = list(range(50))
for i in range(50):

	for h in [horizon_sizes]:

		if i in feasibility_total[h]:

			task_examples.remove(i)
			break

print(task_examples)


#Load the json data
with open('examples/wip/tasho_paper_examples/p2p_L1_terminal_cost.txt', 'r') as fp:
	results = json.load(fp)

