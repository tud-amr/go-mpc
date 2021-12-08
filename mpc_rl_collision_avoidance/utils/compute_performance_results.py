import numpy as np

def process_statistics(episode_stats):
    # Compute Statistics
    time_to_goal = 0.0
    n_collisions = 0
    n_deadlocks = 0
    avg_traveled_distance = 0.0
    for traj_id in range(len(episode_stats)):

        n_collisions += episode_stats[traj_id]["collision"]
        n_deadlocks += episode_stats[traj_id]["stuck"]
        if not(episode_stats[traj_id]["collision"]) or not(episode_stats[traj_id]["stuck"]):
            traveled_distance = 0.0
            time_to_goal += episode_stats[traj_id]["time_to_goal"][0] / len(episode_stats)
            for t in range(1,episode_stats[traj_id]["ego_agent_traj"].shape[0],1):
                traveled_distance += np.linalg.norm(np.array([episode_stats[traj_id]["ego_agent_traj"][t,1]-episode_stats[traj_id]["ego_agent_traj"][t-1,1],
                                                       episode_stats[traj_id]["ego_agent_traj"][t,2]-episode_stats[traj_id]["ego_agent_traj"][t-1,2]]))
            avg_traveled_distance += traveled_distance / len(episode_stats)

    print("***********Performance Results**********************")
    print("***********Number of Collisions**********************")
    print("***********       "+str(n_collisions)+"           **********************")
    print("***********Number of Deadlocks**********************")
    print("***********       " + str(n_deadlocks) + "           **********************")
    print("***********Traveled Distance**********************")
    print("***********       " + str(avg_traveled_distance) + "           **********************")
    print("***********Time to goal**********************")
    print("***********       " + str(time_to_goal) + "           **********************")

    return [{
        "n_collisions": float(n_collisions),
        "n_deadlocks": n_deadlocks,
        "avg_traveled_distance": float(avg_traveled_distance),
        "time_to_goal": float(time_to_goal),
    }]