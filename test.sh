# How to test GO-MPC in multi-agent scenario
#python test_homogeneous.py --policy MPCRLPolicy --n-agents 3 --exp-id 1 --scenario homogeneous_agents_swap --n-episodes 100;

# How to test GA3CCADRLPolicy
#python test_ga3c.py --exp-id 1 --policy GA3CCADRLPolicy --n-agents 3 --n-episodes 100;

# How to test just MPC controller
#python test-mpc.py --exp-id 1 --n-agents 5 --n-episodes 100;

# How to test GA3CCADRLPolicy for multi-agent scenario
#python test_homogeneous.py --policy GA3CCADRLPolicy --n-agents 5 --exp-id 1 --scenario homogeneous_agents_swap --n-episodes 100;

# How to test GO-MPC
python test.py --policy MPCRLPolicy --n-agents 5 --exp-id 1 --n-episodes 100 --scenario train_agents_swap_circle;

# To record add --record flag
python test.py --policy MPCRLPolicy --n-agents 5 --exp-id 1 --record --n-episodes 100 --scenario train_agents_swap_circle;