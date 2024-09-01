# importing libraries
import energyflow as ef
import numpy as np
from energyflow.datasets import mod

# specify the path to the h5 file
FILEPATH = "/home/anuranja/sanmay_project/work/toy_jet_classification/dataset/MOD/CMS2011AJets/JetPrimaryDataset/h5/"

# specify the filename of the dataset file
FILENAME = "CMS_Jet300_pT375-infGeV.h5"


def import_CMS2011AJets_dataset(
    path=FILEPATH,
    pt_max=550,
    pt_min=500,
    eta_max=1.9,
    hadrons_type="None",
):

    # Load the dataset
    if hadrons_type == "charged":

        charged_hadrons_dataset = mod.MODDataset(
            FILENAME,
            f"jet_pts > {pt_min} & jet_pts < {pt_max} & abs_jet_eta < {eta_max}",
            path=path,
        )

        for i in range(len(charged_hadrons_dataset)):

            charged_hadrons_mask = mod.filter_particles(
                charged_hadrons_dataset.pfcs[i],
                which="charged",
                pt_cut=1.0,
                chs=True
            )
            charged_hadrons_temp = charged_hadrons_dataset.pfcs[i][charged_hadrons_mask]
            charged_hadrons_temp = np.hstack((charged_hadrons_temp[:, :3], ef.pids2chrgs(charged_hadrons_temp[:, 4]).reshape(-1, 1)))
            charged_hadrons_dataset.pfcs[i] = charged_hadrons_temp

        charged_hadrons_particles = np.array([arr for arr in charged_hadrons_dataset.pfcs if arr.size != 0], dtype=object)
        return charged_hadrons_particles

    if hadrons_type == "all":

        all_hadrons_dataset = mod.MODDataset(
            FILENAME,
            f"jet_pts > {pt_min} & jet_pts < {pt_max} & abs_jet_eta < {eta_max}",
            path=path,
        )

        for i in range(len(all_hadrons_dataset)):

            all_hadrons_mask = mod.filter_particles(
                all_hadrons_dataset.pfcs[i],
                which="all",
                pt_cut=1.0,
                chs=true
            )
            all_hadrons_temp = all_hadrons_dataset.pfcs[i][all_hadrons_mask]
            all_hadrons_temp = all_hadrons_temp[:, :3]
            all_hadrons_dataset.pfcs[i] = all_hadrons_temp

        all_hadrons_particles = np.array([arr for arr in all_hadrons_dataset.pfcs if arr.size != 0], dtype=object)
        return all_hadrons_particles

    if hadrons_type == "neutral":

        neutral_hadrons_dataset = mod.MODDataset(
            FILENAME,
            f"jet_pts > {pt_min} & jet_pts < {pt_max} & abs_jet_eta < {eta_max}",
            path=path,
        )

        for i in range(len(neutral_hadrons_dataset)):

            neutral_hadrons_mask = mod.filter_particles(
                neutral_hadrons_dataset.pfcs[i],
                which="neutral",
                pt_cut=1.0,
                chs=True
            )
            neutral_hadrons_temp = neutral_hadrons_dataset.pfcs[i][neutral_hadrons_mask]
            neutral_hadrons_temp = neutral_hadrons_temp[:, :3]
            neutral_hadrons_dataset.pfcs[i] = neutral_hadrons_temp

        neutral_hadrons_particles = np.array([arr for arr in neutral_hadrons_dataset.pfcs if arr.size != 0], dtype=object)
        return neutral_hadrons_particles


# hadron_dataset = import_CMS2011AJets_dataset(
#     path=FILEPATH,
#     pt_max=550,
#     pt_min=500,
#     eta_max=1.9,
#     hadrons_type="charged",
# )
# print(hadron_dataset)
