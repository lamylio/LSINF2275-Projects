import matplotlib.pyplot as plt
def plot_train(scores, mean_scores, epsilons) :
    plt.clf()    
    plt.xlabel('Number of games')
    plt.ylabel('Score', color="tab:blue")
    plt.tick_params(axis="y", color="tab:blue")
    plt.plot(scores, color="tab:blue")
    plt.plot(mean_scores, color="tab:orange")
    plt.text(1.01*len(mean_scores), mean_scores[-1], str(round(mean_scores[-1])))
    ax2 = plt.twinx()
    ax2.plot(epsilons, color="tab:red", alpha=0.5)
    ax2.set_ylabel("Epsilon", color="tab:red")
    plt.show(block=False)
    plt.pause(100)


def plot_test(scores, mean_scores) :
    plt.clf()
    plt.xlabel('Number of games')
    plt.ylabel('Score', color="tab:blue")
    plt.tick_params(axis="y", color="tab:blue")
    plt.plot(scores, color="tab:blue")
    plt.plot(mean_scores, color="tab:orange")
    plt.text(1.01*len(mean_scores), mean_scores[-1], str(round(mean_scores[-1])))
    plt.show(block=False)
    plt.pause(100)

# ====================================================================

import json
with open("./resources/json/test_results_2_smooth_30k.json") as f:
    RESULTS = json.load(f)

# plot_train(RESULTS["SCORES"], RESULTS["SCORE_MEANS"], RESULTS["EPSILONS"])
plot_test(RESULTS["SCORES"], RESULTS["SCORE_MEANS"])