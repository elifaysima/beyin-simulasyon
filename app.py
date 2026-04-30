from flask import Flask, jsonify, request, Response, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import os
 
app = Flask(__name__)
CORS(app)
 
REGIONS = {
    "V1_L": {"name":"Primary Visual Cortex L", "lobe":"occipital", "coords":[12,-85,3], "base_color":[0.2,0.6,0.9]},
    "V1_R": {"name":"Primary Visual Cortex R", "lobe":"occipital", "coords":[-12,-85,3], "base_color":[0.2,0.6,0.9]},
    "V2_L": {"name":"Secondary Visual Cortex L", "lobe":"occipital", "coords":[20,-78,10], "base_color":[0.3,0.5,1.0]},
    "A1_L": {"name":"Primary Auditory Cortex L", "lobe":"temporal", "coords":[46,-22,10], "base_color":[0.9,0.5,0.2]},
    "A1_R": {"name":"Primary Auditory Cortex R", "lobe":"temporal", "coords":[-46,-22,10], "base_color":[0.9,0.5,0.2]},
    "Broca_L": {"name":"Broca's Area", "lobe":"frontal", "coords":[42,22,8], "base_color":[0.1,0.8,0.3]},
    "Wernicke_L": {"name":"Wernicke's Area", "lobe":"temporal", "coords":[50,-38,18], "base_color":[0.3,0.7,0.2]},
    "M1_L": {"name":"Primary Motor Cortex L", "lobe":"frontal", "coords":[38,-23,55], "base_color":[0.4,0.8,0.4]},
    "M1_R": {"name":"Primary Motor Cortex R", "lobe":"frontal", "coords":[-38,-23,55], "base_color":[0.4,0.8,0.4]},
    "S1_L": {"name":"Somatosensory Cortex L", "lobe":"parietal", "coords":[42,-24,50], "base_color":[0.5,0.9,0.5]},
    "DLPFC_L": {"name":"Dorsolateral Prefrontal Ctx L", "lobe":"frontal", "coords":[38,36,28], "base_color":[0.9,0.2,0.2]},
    "DLPFC_R": {"name":"Dorsolateral Prefrontal Ctx R", "lobe":"frontal", "coords":[-38,36,28], "base_color":[0.9,0.2,0.2]},
    "vmPFC": {"name":"Ventromedial Prefrontal Ctx", "lobe":"frontal", "coords":[0,48,-7], "base_color":[0.8,0.3,0.5]},
    "ACC": {"name":"Anterior Cingulate Cortex", "lobe":"frontal", "coords":[0,32,18], "base_color":[0.7,0.5,0.8]},
    "Hippocampus_L": {"name":"Hippocampus L", "lobe":"temporal", "coords":[26,-22,-12], "base_color":[0.5,0.2,0.7]},
    "Hippocampus_R": {"name":"Hippocampus R", "lobe":"temporal", "coords":[-26,-22,-12], "base_color":[0.5,0.2,0.7]},
    "Amygdala_L": {"name":"Amygdala L", "lobe":"temporal", "coords":[22,-4,-18], "base_color":[1.0,0.4,0.0]},
    "Amygdala_R": {"name":"Amygdala R", "lobe":"temporal", "coords":[-22,-4,-18], "base_color":[1.0,0.4,0.0]},
    "Thalamus_L": {"name":"Thalamus L", "lobe":"diencephalon", "coords":[10,-16,6], "base_color":[0.6,0.6,0.1]},
    "Thalamus_R": {"name":"Thalamus R", "lobe":"diencephalon", "coords":[-10,-16,6], "base_color":[0.6,0.6,0.1]},
    "SN_L": {"name":"Substantia Nigra L (Dopamin)", "lobe":"midbrain", "coords":[10,-18,-8], "base_color":[0.2,0.9,0.2]},
    "Raphe": {"name":"Raphe Nuclei (Serotonin)", "lobe":"brainstem", "coords":[0,-26,-20], "base_color":[0.9,0.9,0.2]},
    "LC": {"name":"Locus Coeruleus (Noradrenalin)", "lobe":"pons", "coords":[0,-34,-28], "base_color":[0.2,0.4,1.0]},
    "CC_Genu": {"name":"Corpus Callosum (Genu)", "lobe":"commissure", "coords":[0,20,12], "base_color":[0.95,0.90,0.85]},
    "CC_Body": {"name":"Corpus Callosum (Body)", "lobe":"commissure", "coords":[0,0,20], "base_color":[0.92,0.88,0.82]},
    "CC_Splenium": {"name":"Corpus Callosum (Splenium)", "lobe":"commissure", "coords":[0,-28,18], "base_color":[0.90,0.86,0.80]},
    "BA47_L": {"name":"Inferior Frontal Gyrus (Yabancı Dil)", "lobe":"frontal", "coords":[44,26,-4], "base_color":[0.2,0.85,0.6]},
    "VLPFC_L": {"name":"Ventrolateral PFC (Dil Kontrolü)", "lobe":"frontal", "coords":[46,30,8], "base_color":[0.2,0.75,0.5]},
}
 
CONNECTIONS = [
    ["V1_L","V2_L",0.9], ["V1_R","V2_L",0.3],
    ["A1_L","Wernicke_L",0.8], ["Wernicke_L","Broca_L",0.7],
    ["M1_L","S1_L",0.9], ["M1_R","S1_L",0.2],
    ["DLPFC_L","ACC",0.6], ["DLPFC_R","ACC",0.6],
    ["ACC","Amygdala_L",0.5], ["ACC","Amygdala_R",0.5],
    ["Hippocampus_L","Amygdala_L",0.8], ["Hippocampus_R","Amygdala_R",0.8],
    ["Thalamus_L","V1_L",0.7], ["Thalamus_R","V1_R",0.7],
    ["SN_L","DLPFC_L",0.4], ["Raphe","Hippocampus_L",0.5], ["LC","Amygdala_L",0.6],
    ["CC_Genu","DLPFC_L",0.9], ["CC_Genu","DLPFC_R",0.9],
    ["CC_Body","M1_L",0.9], ["CC_Body","M1_R",0.9], ["CC_Body","S1_L",0.85],
    ["CC_Splenium","V1_L",0.9], ["CC_Splenium","V1_R",0.9], ["CC_Splenium","Hippocampus_L",0.8], ["CC_Splenium","Hippocampus_R",0.8],
    ["BA47_L","Broca_L",0.85], ["BA47_L","Wernicke_L",0.75], ["VLPFC_L","DLPFC_L",0.7], ["VLPFC_L","BA47_L",0.8],
]
 
BASE_ACTIVITY = {r: 0.5 for r in REGIONS}
 
MOD_PRESETS = {
    "relaxed":      {"V1_L":0.4,"V1_R":0.4,"A1_L":0.4,"Broca_L":0.3,"M1_L":0.3,"DLPFC_L":0.5,"Amygdala_L":0.3,"Hippocampus_L":0.4,"Thalamus_L":0.4,"SN_L":0.5,"Raphe":0.6,"LC":0.2},
    "crowd":        {"A1_L":0.85,"A1_R":0.85,"V1_L":0.6,"Amygdala_L":0.9,"Amygdala_R":0.9,"ACC":0.8,"DLPFC_L":0.7,"LC":0.7},
    "morning_wake": {"Thalamus_L":0.7,"Thalamus_R":0.7,"LC":0.8,"DLPFC_L":0.2,"V1_L":0.3,"A1_L":0.2,"SN_L":0.3,"Raphe":0.3},
    "music":        {"A1_L":0.95,"A1_R":0.95,"Hippocampus_L":0.7,"Amygdala_L":0.7,"V1_L":0.3,"DLPFC_L":0.4,"ACC":0.5,"Raphe":0.9},
    "fear":         {"Amygdala_L":1.0,"Amygdala_R":1.0,"LC":1.0,"Thalamus_L":0.9,"V1_L":0.8,"ACC":0.7,"DLPFC_L":0.2,"Hippocampus_L":0.8},
    "active_study": {"DLPFC_L":0.95,"DLPFC_R":0.95,"ACC":0.7,"Broca_L":0.8,"Hippocampus_L":0.6,"Amygdala_L":0.2,"LC":0.3},
    "depressed":    {"DLPFC_L":0.2,"DLPFC_R":0.2,"vmPFC":0.3,"ACC":0.4,"Amygdala_L":0.3,"Hippocampus_L":0.3,"Raphe":0.3,"SN_L":0.3},
    "hungry":       {"Hippocampus_L":0.5,"Amygdala_L":0.7,"vmPFC":0.6,"Thalamus_L":0.6,"LC":0.6},
    "sleepy":       {"Thalamus_L":0.3,"DLPFC_L":0.2,"ACC":0.3,"LC":0.4,"V1_L":0.2,"A1_L":0.2},
    "angry":        {"Amygdala_L":1.0,"Amygdala_R":1.0,"DLPFC_L":0.3,"ACC":0.8,"LC":0.9,"M1_L":0.9},
    "love":         {"vmPFC":0.9,"ACC":0.8,"Amygdala_L":0.7,"Hippocampus_L":0.8,"SN_L":0.9},
    "meditation":   {"DLPFC_L":0.8,"ACC":0.9,"Thalamus_L":0.3,"Amygdala_L":0.2,"Raphe":0.9},
    "drunk":        {"DLPFC_L":0.2,"ACC":0.3,"M1_L":0.3,"Hippocampus_L":0.4,"Thalamus_L":0.5},
    "exercise":     {"M1_L":1.0,"M1_R":1.0,"DLPFC_L":0.7,"LC":0.9,"SN_L":0.8},
    "creative":     {"DLPFC_L":0.8,"ACC":0.7,"Hippocampus_L":0.7,"vmPFC":0.8,"Raphe":0.7},
    "happy":        {"vmPFC":0.9,"ACC":0.7,"SN_L":0.9,"Raphe":0.8,"DLPFC_L":0.7,"Amygdala_L":0.6},
    "sad":          {"ACC":0.7,"Amygdala_L":0.7,"Hippocampus_L":0.5,"Raphe":0.3,"DLPFC_L":0.3,"vmPFC":0.4},
    "jealous":      {"Amygdala_L":0.85,"ACC":0.8,"DLPFC_L":0.5,"vmPFC":0.4,"LC":0.7},
    "grief":        {"Amygdala_L":0.9,"ACC":0.8,"Hippocampus_L":0.7,"Raphe":0.2,"DLPFC_L":0.2,"vmPFC":0.3},
    "excited":      {"SN_L":0.95,"DLPFC_L":0.8,"ACC":0.7,"Amygdala_L":0.7,"LC":0.8,"vmPFC":0.8},
    "nostalgic":    {"Hippocampus_L":0.9,"Hippocampus_R":0.9,"vmPFC":0.7,"Raphe":0.7,"Amygdala_L":0.6},
    "lonely":       {"ACC":0.8,"Amygdala_L":0.7,"DLPFC_L":0.3,"Raphe":0.3,"vmPFC":0.3},
    "proud":        {"vmPFC":0.85,"DLPFC_L":0.8,"SN_L":0.85,"ACC":0.6},
    "shame":        {"ACC":0.9,"Amygdala_L":0.8,"DLPFC_L":0.4,"vmPFC":0.3,"Raphe":0.4},
    "disgust":      {"Amygdala_L":0.85,"ACC":0.7,"V1_L":0.5,"A1_L":0.5},
    "awe":          {"ACC":0.8,"vmPFC":0.9,"DLPFC_L":0.7,"Hippocampus_L":0.7,"Thalamus_L":0.8},
    "problem_solving":{"DLPFC_L":0.95,"DLPFC_R":0.9,"ACC":0.8,"Hippocampus_L":0.7,"Thalamus_L":0.7},
    "reading":      {"Wernicke_L":0.9,"Broca_L":0.7,"DLPFC_L":0.6,"V1_L":0.7,"V2_L":0.7},
    "daydreaming":  {"vmPFC":0.8,"Hippocampus_L":0.8,"ACC":0.5,"DLPFC_L":0.3},
    "flow":         {"DLPFC_L":0.85,"ACC":0.6,"SN_L":0.8,"Raphe":0.7,"Amygdala_L":0.2,"LC":0.3},
    "hyperfocus":   {"DLPFC_L":1.0,"ACC":0.8,"Thalamus_L":0.9,"LC":0.8,"Amygdala_L":0.1},
    "mind_wandering":{"vmPFC":0.9,"Hippocampus_L":0.9,"ACC":0.4,"DLPFC_L":0.2},
    "decision_making":{"DLPFC_L":0.9,"vmPFC":0.8,"ACC":0.85,"Amygdala_L":0.6,"Thalamus_L":0.7},
    "multitasking": {"DLPFC_L":0.9,"DLPFC_R":0.9,"ACC":0.85,"Thalamus_L":0.8,"LC":0.7},
    "public_speaking":{"Broca_L":0.95,"M1_L":0.7,"Amygdala_L":0.8,"ACC":0.8,"DLPFC_L":0.7,"LC":0.8},
    "social_anxiety":{"Amygdala_L":0.95,"Amygdala_R":0.95,"ACC":0.8,"DLPFC_L":0.3,"LC":0.85},
    "empathy":      {"vmPFC":0.9,"ACC":0.85,"Amygdala_L":0.7,"Hippocampus_L":0.6,"Raphe":0.8},
    "conflict":     {"Amygdala_L":0.9,"ACC":0.85,"DLPFC_L":0.5,"LC":0.85,"M1_L":0.6},
    "intimacy":     {"vmPFC":0.9,"Hippocampus_L":0.8,"SN_L":0.9,"Raphe":0.85,"Amygdala_L":0.6},
    "anxious":      {"Amygdala_L":0.95,"Amygdala_R":0.9,"LC":0.95,"DLPFC_L":0.3,"ACC":0.85,"Thalamus_L":0.8},
    "panic":        {"Amygdala_L":1.0,"Amygdala_R":1.0,"LC":1.0,"Thalamus_L":0.95,"M1_L":0.8,"DLPFC_L":0.1},
    "manic":        {"DLPFC_L":0.9,"SN_L":0.95,"ACC":0.9,"Amygdala_L":0.5,"vmPFC":0.9,"LC":0.85},
    "dissociation": {"DLPFC_L":0.2,"Thalamus_L":0.3,"ACC":0.4,"Hippocampus_L":0.3,"Amygdala_L":0.2},
    "ocd_loop":     {"ACC":0.95,"DLPFC_L":0.6,"Thalamus_L":0.85,"Amygdala_L":0.7,"Raphe":0.3},
    "ptsd_trigger": {"Amygdala_L":1.0,"Amygdala_R":1.0,"Hippocampus_L":0.9,"LC":1.0,"DLPFC_L":0.1,"ACC":0.8},
    "burnout":      {"DLPFC_L":0.2,"DLPFC_R":0.2,"Raphe":0.2,"SN_L":0.2,"ACC":0.3,"Hippocampus_L":0.3},
    "adhd_scatter": {"DLPFC_L":0.25,"vmPFC":0.8,"LC":0.85,"ACC":0.5,"Amygdala_L":0.6},
    "thirsty":      {"Thalamus_L":0.8,"Thalamus_R":0.8,"Hippocampus_L":0.5,"LC":0.6},
    "pain":         {"S1_L":0.95,"Thalamus_L":0.9,"ACC":0.9,"Amygdala_L":0.8,"DLPFC_L":0.4},
    "pleasure":     {"SN_L":1.0,"vmPFC":0.9,"ACC":0.7,"Raphe":0.85,"Amygdala_L":0.6},
    "orgasm":       {"SN_L":1.0,"vmPFC":0.95,"Raphe":0.9,"Hippocampus_L":0.7,"Amygdala_L":0.7,"LC":0.8},
    "caffeine":     {"LC":0.9,"DLPFC_L":0.8,"ACC":0.7,"Thalamus_L":0.8,"SN_L":0.7},
    "cold_shower":  {"LC":0.95,"Thalamus_L":0.8,"S1_L":0.9,"DLPFC_L":0.7,"Raphe":0.7},
    "fever":        {"Thalamus_L":0.85,"Thalamus_R":0.85,"Hippocampus_L":0.4,"DLPFC_L":0.3,"ACC":0.5},
    "bored":        {"DLPFC_L":0.3,"ACC":0.4,"vmPFC":0.6,"Raphe":0.4,"SN_L":0.3},
    "yawning":      {"Thalamus_L":0.4,"DLPFC_L":0.2,"Raphe":0.4,"LC":0.35},
    "n1_sleep":     {"Thalamus_L":0.4,"Thalamus_R":0.4,"DLPFC_L":0.15,"ACC":0.2,"LC":0.2,"Raphe":0.5,"V1_L":0.1,"A1_L":0.15},
    "n2_sleep":     {"Thalamus_L":0.3,"Thalamus_R":0.3,"DLPFC_L":0.1,"ACC":0.15,"LC":0.1,"Raphe":0.4,"Hippocampus_L":0.5,"SN_L":0.2},
    "n3_sleep":     {"Thalamus_L":0.15,"Thalamus_R":0.15,"DLPFC_L":0.05,"ACC":0.1,"LC":0.05,"Raphe":0.3,"SN_L":0.15,"Hippocampus_L":0.3,"Amygdala_L":0.1,"V1_L":0.05,"A1_L":0.05},
    "rem_sleep":    {"V1_L":0.7,"V1_R":0.7,"V2_L":0.65,"Amygdala_L":0.85,"Amygdala_R":0.85,"Hippocampus_L":0.9,"Hippocampus_R":0.9,"ACC":0.6,"vmPFC":0.7,"DLPFC_L":0.1,"LC":0.05,"Raphe":0.2},
    "lucid_dream":  {"V1_L":0.8,"V1_R":0.8,"DLPFC_L":0.75,"DLPFC_R":0.75,"ACC":0.8,"Hippocampus_L":0.85,"Amygdala_L":0.6,"vmPFC":0.7,"Thalamus_L":0.6},
    "sleep_paralysis":{"Amygdala_L":1.0,"Amygdala_R":1.0,"LC":0.9,"Thalamus_L":0.5,"M1_L":0.05,"M1_R":0.05,"V1_L":0.6,"DLPFC_L":0.2,"ACC":0.7},
    "hypnagogia":   {"Thalamus_L":0.45,"Thalamus_R":0.45,"V1_L":0.4,"Hippocampus_L":0.55,"Raphe":0.55,"LC":0.25,"DLPFC_L":0.2,"ACC":0.3},
    "foreign_speaking": {"Broca_L":0.95,"BA47_L":0.95,"VLPFC_L":0.9,"Wernicke_L":0.8,"DLPFC_L":0.85,"ACC":0.8,"Hippocampus_L":0.75,"LC":0.7,"M1_L":0.6,"CC_Genu":0.85},
    "foreign_listening": {"Wernicke_L":0.95,"A1_L":0.9,"A1_R":0.7,"BA47_L":0.8,"Hippocampus_L":0.7,"DLPFC_L":0.7,"ACC":0.65,"Thalamus_L":0.75},
}
 
def wilson_cowan_step(activities, connections, steps=50, dt=0.1):
    A = np.array(list(activities.values()))
    reg_list = list(activities.keys())
    n = len(reg_list)
    W = np.zeros((n, n))
    for src, tgt, w in connections:
        if src in reg_list and tgt in reg_list:
            i, j = reg_list.index(src), reg_list.index(tgt)
            W[i, j] = w * 0.5
    tau = 1.0
    for _ in range(steps):
        x = W @ A
        F = 1.0 / (1.0 + np.exp(-x))
        dA = (-A + F) / tau
        A += dA * dt
        A = np.clip(A, 0.0, 1.0)
    return {reg_list[i]: float(A[i]) for i in range(n)}
 
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/brain.glb')
def serve_brain():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'brain.glb')
 
@app.route('/personalize', methods=['POST'])
def personalize():
    data = request.json
    activities = BASE_ACTIVITY.copy()
    hand = data.get('handedness', 'right')
    if hand == 'left':
        activities['Broca_L'] += 0.2
        activities['M1_R'] = activities.get('M1_R', 0.5) + 0.15
    elif hand == 'ambidextrous':
        activities['Broca_L'] += 0.1
        activities['M1_L'] += 0.1
        activities['M1_R'] = activities.get('M1_R', 0.5) + 0.1
 
    myopia = float(data.get('myopia_now', 0))
    years  = int(data.get('myopia_years', 0))
    if years > 0:
        activities['V1_L'] -= 0.02 * myopia * years
        activities['V2_L'] -= 0.01 * myopia * years
    elif myopia > 0:
        activities['V1_L'] -= 0.1 * myopia
        activities['V2_L'] -= 0.05 * myopia
 
    asd_score   = float(data.get('asd_score', 0))
    adhd_score  = float(data.get('adhd_score', 0))
    dep_score   = float(data.get('dep_score', 0))
    anx_score   = float(data.get('anx_score', 0))
    manic_score = float(data.get('manic_score', 0))
 
    if asd_score > 50:
        f = (asd_score - 50) / 50
        activities['Amygdala_L'] += 0.2 * f
        activities['ACC']        -= 0.1 * f
        activities['DLPFC_L']    -= 0.1 * f
    if adhd_score > 50:
        f = (adhd_score - 50) / 50
        activities['DLPFC_L'] -= 0.25 * f
        activities['vmPFC']   += 0.2  * f
        activities['LC']      += 0.2  * f
    if dep_score > 50:
        f = (dep_score - 50) / 50
        activities['DLPFC_L']    -= 0.3 * f
        activities['ACC']        -= 0.2 * f
        activities['Amygdala_L'] += 0.1 * f
    if anx_score > 50:
        f = (anx_score - 50) / 50
        activities['Amygdala_L'] += 0.3 * f
        activities['LC']         += 0.3 * f
        activities['DLPFC_L']    -= 0.2 * f
    if manic_score > 50:
        f = (manic_score - 50) / 50
        activities['DLPFC_L']    += 0.3 * f
        activities['ACC']        += 0.2 * f
        activities['Amygdala_L'] -= 0.2 * f
 
    for k in activities:
        activities[k] = max(0.0, min(1.0, activities[k]))
 
    activities = wilson_cowan_step(activities, CONNECTIONS, steps=30, dt=0.05)
    return jsonify({"activities": activities, "regions": REGIONS, "connections": CONNECTIONS})
 
@app.route('/set_mode', methods=['POST'])
def set_mode():
    mode = request.json.get('mode', 'relaxed')
    acts = BASE_ACTIVITY.copy()
    acts.update(MOD_PRESETS.get(mode, {}))
    acts = wilson_cowan_step(acts, CONNECTIONS, steps=20, dt=0.1)
    return jsonify({"activities": acts, "mode": mode, "regions": REGIONS, "connections": CONNECTIONS})
 
if __name__ == '__main__':
    app.run(debug=True, port=5002, use_reloader=False)