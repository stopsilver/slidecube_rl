---Environment---
slidesquare_env.py
	Slidesquare environment
SlideSquare_env_test.py
	how to use environment

---Tabular method for SlideSquare---
How to run:
SlideSquare_policy_statevalue_solve.py
	Solution of 3x3 slidesquare environment using policy evaluation-improvement method
	Result :
		statevalue_3x3.txt - state-value table for all states
		policy_3x3.txt - policy table for all states
SlideSquare_solve.py
	solution demo for random scramble (uses policy_3x3.txt)
	
---NNet method for SlideSquare---
NNet_SlideSquare_train.py
	Solution of 3x3 slidesquare environment using policy evaluation-improvement method
	Two NNets used : model_val - state-value approximation, model_act - policy estimation ('softmax')
	It could be regarded as actor-critic method
	Result :
		sl_stval_3x3.h5 - NNet for state-value
		sl_act_3x3.h5 - NNet for policy
		sl_3x3.txt - state-value table for all states - compare it to statevalue_3x3.txt using natlab script below
		-----------------------------------------
		a=load('statevalue_3x3.txt');
		b=load('sl_3x3.txt');
		plot([a b]);
		-----------------------------------------
NNet_SlideSquare_solve.py
	solution demo for random scramble (uses sl_act_3x3.h5)
	
======SlideCube=========
NNet_SlideCube_train.py
		Parameters :
	SN=2 # or 3 - Size of SlideCube
	gamma=1      # currently used 1
	ep=0.       <-- use 0 # set ep(silon) > 0 for epsilon gradient policy (not effective)
	use_weight=False	# there are two types of training  - with/without distance weight (applies less weight for more distanced state from goal)
	train_batch_size=250
	train_buf_size=1000
	train_scramble_depth=15		# 15 for 2x2; 20 for 3x3
		Output :
	generates model file (ex. "slcube_stval_2x2_pure.h5")

NNet_SlideCube_statevalue_model_test.py
		Parameters :
	SN=2 # or 3 - Size of SlideCube
	gamma=1      # currently used 1
	statevalue_policy=True         # True - state-value policy, action-value policy
	ep=0.       <-- use 0 # set ep(silon) > 0 for epsilon gradient policy (not effective)
	use_weight=False	# there are two types of training  - with/without distance weight (applies less weight for more distanced state from goal)
	s=cube_env.scramble_cube(4)		# scramble cube by number of moves
		Output :
	prints solution in command line
	generates files "startpos.txt", "solutionmoves.txt"
	----------------------------------
		run solution visualization in matlab:
		Visualize_solution.m
	----------------------------------
		