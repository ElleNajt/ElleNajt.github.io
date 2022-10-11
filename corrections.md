
## Corrections: 

Oct 22 - Presenting at the [AMS Special Session on Partitioning and Redistricting, I](https://meetings.ams.org/math/fall2022w/meetingapp.cgi/Paper/17358). 

I would like to note that the abstract I submitted contains, in my opinion, a serious error, since I cannot correct this error on the AMS webpage,  here is a corrected abstract: 

> Ensemble methods for identifying ’gerrymanders’ have become increasingly influential over the past decade, but the robustness of these methods to extraneous factors has been understudied. Ensemble methods work by building a collection of redistricting plans, and examining the properties of a proposed plan in relation to this collection. In (N., DeFord, Solomon), we investigated the sensitivity of several of these methods to the essentially arbitrary way that voting tabulation districts (VTDs) partition simplified political geography, and found that infinitesimal perturbations of VTD geometry lead to meaningfully different ensemble averages of the number of seats won by a particular party, a phenomenon that we called metamandering. These findings posed a challenge for the meaningfulness of the ’average seats won’ ensemble statistic and, therefore, its practical utility; however, it remained unclear to what extent similar examples existed on real data. I will report on my progress with Karrmann, Mark and Zhang on finding metamanders in real world data.

To highlight the difference from the old abstract:

While I still believe that it is unclear whether these distributions over connected partitions refer to an observable reality, and that this is supported by their apparent non-robustness to infinitsemal pertubations to VTDs of the grid graph that we showed in (N., DeFord, Solomon), it is not true that I was able to find meaningful examples of comparable complexity to real world data that used infinitesimal pertubations in the same way as the toy examples. 

My mistakes were:
1. I did not appropriately measure the signal-to-noise ratio of  experiment, and this lead me to be overconfident that we had succesfully replicated 'metamandering' on real-world data.
2. I was overconfident about the richness of the search space for the strategy we developed to generalize metamandering from (N., DeFord, Solomon), and this lead to me to require less evidence for a meaningful shift than I should have. After sitting with this I think the particular approach we were taking will not generalize our previous toy examples; I will explain the obstruction.

I remain confident that the fundamental message from (N., DeFord, Solomon) is sound, and while my mistake weakens the message in a way that necessitates correction, my opinion remains that real-world data 'metamandering' is possible. We are exploring a promising alternative path towards real-data metamandering, which I will report progress on. 

Moreover, from a more philosophical point of view, my opinion is that the experiments from (N., DeFord, Solomon) independently make a strong case that we don't have a clear understanding of what ensemble methods are measuring, or whether they define a legitimate baseline. In my mind this is an independent question from whether metamandering attacks are feasible on real-world data.
