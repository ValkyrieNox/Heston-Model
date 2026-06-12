import json
P="/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel/training"
def show(label, path):
    try:
        d=json.load(open(path))
        mc=d.get("model_config",{}); tc=d.get("train_config",{})
        print("--- %s ---"%label)
        print("  stage=%s hidden=%s blocks=%s cond=%s tdim=%s"%(
            d.get("stage"), mc.get("hidden_dim"), mc.get("num_blocks"),
            mc.get("condition_dim"), mc.get("time_embedding_dim")))
        print("  bs=%s ep=%s lr=%s sched_ss=%s adrop=%s lwfm=%s"%(
            tc.get("batch_size"), tc.get("epochs"), tc.get("lr"),
            tc.get("scheduled_sampling_max_prob"), tc.get("action_dropout_prob"),
            tc.get("lambert_w_delta")))
    except Exception as e:
        print(label,"ERR",e)
show("FM vol", P+"/vol_fm/vol_p3_full_parallel/config.json")
show("FM ret", P+"/ret_fm/ret_p3_full_parallel/config.json")
show("B1 ret", P+"/B_0602/B1_sched08_ret/config.json")
show("strong vol", P+"/b_partner_recipe_teacher_0602/b_partner_recipe_vol_256x6_bs8192_0602/config.json")
show("strong ret", P+"/b_partner_recipe_teacher_0602/b_partner_recipe_ret_256x6_bs8192_sched_0602/config.json")
show("LWFM d0.05 vol", P+"/vol_lwfm_d0.05/vol_lwfm_d0.05/config.json")
show("CD vol", P+"/cd_vol/cd_vol_p3_full_parallel/config.json")
show("CD ret", P+"/cd_ret/cd_ret_p3_full_parallel/config.json")
show("MF ret", P+"/mf_ret/mf_ret_p3_full_parallel/config.json")
show("combined", P+"/combined_0602/combined_moment_sigmmd_energy_sigw1_bs512_10ep_0602/config.json")
