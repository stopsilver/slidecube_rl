# slidecube_rl

---Environment---
slidesquare_env.py
	Slidesquare environment
SlideSquare_env_test.py
	how to use environment

---Tabular method---
How to run:
SlideSquare_policy_statevalue_solve.py
	Solution of 3x3 slidesquare environment using policy evaluation-improvement method
	Result :
		statevalue_3x3.txt - state-value table for all states
		policy_3x3.txt - policy table for all states
SlideSquare_solve.py
	solution demo for random scramble (uses policy_3x3.txt)
	
---NNet method---
NNet_train.py
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
