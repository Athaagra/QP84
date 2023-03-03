# QP84

This project is under construction(Bugs not yet for testing)!<\br>

<h2>One environment</h2>
<p>Classical channel ql: 1bit:0.54 2bit:0.18 3bit:0.09 4bit:0.0
Quantum channel ql: 1bit:0.49 2bit:0.27 3bit:0.08 4bit:0.00</p>
<p>Classical channel dqn: 1bit:0.58 2bit:0.26 3bit:0.1 4bit:0.01
Quantum channel dqn:1bit:0.44 2bit:0.18 3bit:0.08 4bit:0.09</p>
<p>Classical channel ppo: 1bit:0.5 2bit:0.21 3bit:0.12 4bit:0.05
Quantum channel ppo: 1bit:0.51 2bit:0.25 3bit:0.03 4bit:0.04</p>
<p>Classical channel es: 1bit:0.42 2bit:0.22 3bit:0.13 4bit:0.10
Quantum channel es: 1bit:0.53 2bit:0.28 3bit:0.09 4bit:0.02</p>


<h2>Multiagent Environment</h2>
<p>Classical channel ql:1bit:1.0 2bit:0.72 3bit:0.27 4bit:0.05
Quantum channel ql:1bit:1.0 2bit:0.80 3bit:0.36 4bit:0.19 </p>
<p>Classical channel dqn:1bit:0.74 2bit:0.85 3bit:0.33 4bit:0.16
Quantum channel dqn: 1bit:1.00 2bit:0.73 3bit:0.38 4bit:0.12</p>
<p>Classical channel ppo:1bit:0.54 2bit:0.27 3bit:0.47 4bit:0.19
Quantum channel ppo:1bit:0.61 2bit:0.59 3bit:0.43 4bit:0.23</p>
<p>Classical channel es: 1bit:1.0 2bit:0.82 3bit:0.36 4bit:0.04
Quantum channel es: 1bit:1.0 2bit:0.84 3bit:0.35 4bit:0.05</p>

<h2>Quantum Free Number of Actions</h2>

|      	|                              	| Reward 	| Steps 	|
|------	|------------------------------	|--------	|-------	|
| 1bit 	| Q learning                   	| 0.54   	| 1.0   	|
|      	| Deep q-learning              	| 0.57   	| 5.0   	|
|      	| Proximal policy Optimization 	| 0.46   	| 6.09  	|
|      	| Evolutionary Strategy        	| 0.58   	| 4.36  	|
| 2bit 	| Q-learning                   	| 0.48   	| 1.0   	|
|      	| Deep q-learning              	| 0.55   	| 5.0   	|
|      	| Proximal policy Optimization 	| 0.46   	| 6.09  	|
|      	| Evolutionary Strategy        	| 0.54   	| 4.68  	|
| 3bit 	| Q-learning                   	| 0.4    	| 4.61  	|
|      	| Deep q-learning              	| 0.48   	| 6.0   	|
|      	| Proximal policy Optimization 	| 0.49   	| 5.76  	|
|      	| Evolutionary Strategy        	| 0.52   	| 4.84  	|
| 4bit 	| Q-learning                   	| 0.5    	| 1.0   	|
|      	| Deep q-learning              	| 0.47   	| 5.0   	|
|      	| Proximal Policy Optimization 	| 0.59   	| 5.17  	|
|      	| Evolutionary Strategy        	| 0.46   	| 5.32  	|



|                              | 1bit |     | 2bit |   | 3bit |   | 4bit |   |
|------------------------------|:----:|-----|:----:|---|:----:|---|:----:|---|
| Q-learning                   | 0.51 | 6   | 0.23 | 8 | 0.1  | 8 | 0.06 | 9 |
| DQN                          | 0.5  | 5   | 0.49 | 6 | 0.47 | 5 | 0.5  | 5 |
| Proximal Policy Optimization | 0.55 | 5.5 | 0.30 | 7 | 0.11 | 8 | 0.04 | 9 |
| Evolutionary Strategy        | 0.53 | 6   | 0.23 | 7 | 0    | 9 | 0    | 9 |
