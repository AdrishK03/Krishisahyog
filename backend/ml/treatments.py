# ─────────────────────────────────────────────────────────────────────────────
# treatments.py
# Treatment database — key format: "Plant|<exact_disease_folder_name>"
# Each entry has: severity, cause, symptoms, fertilizer, fungicide, cultural, organic
# ─────────────────────────────────────────────────────────────────────────────

from typing import Optional

TREATMENTS: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════════════
    # POTATO
    # ══════════════════════════════════════════════════════════════════════════
    "Potato|Potato___Early_blight": {
        "severity": "Moderate",
        "cause": "Fungus — Alternaria solani",
        "symptoms": "Dark brown circular spots with concentric rings (target-board pattern) on older/lower leaves. Yellowing around lesions.",
        "fertilizer": "Balanced NPK (19-19-19). Avoid excess nitrogen — promotes lush susceptible growth.",
        "fungicide": [
            "Mancozeb 75 WP — 2.5 g/L water, spray every 7–10 days",
            "Chlorothalonil 75 WP — 2 g/L water",
            "Azoxystrobin 23 SC — 1 mL/L water (systemic, highly effective)",
            "Tebuconazole 25.9 EC — 1 mL/L water",
        ],
        "cultural": [
            "Remove and destroy infected leaves immediately",
            "Avoid overhead irrigation — water at base of plant",
            "Maintain 30 cm row spacing for good airflow",
            "Rotate crops — avoid potato in same field for 2–3 years",
            "Scout weekly; spray preventively before disease appears",
        ],
        "organic": "Neem oil 5 mL/L + baking soda 5 g/L weekly spray. Trichoderma viride 5 g/L as preventive.",
    },

    "Potato|Potato___Late_blight": {
        "severity": "Severe — can destroy entire crop within days",
        "cause": "Water mold — Phytophthora infestans",
        "symptoms": "Water-soaked lesions on leaves turning brown/black rapidly. White fluffy mold on underside in humid weather. Brown rot in tubers.",
        "fertilizer": "Boost potassium using SOP (0-0-50) — strengthens cell walls and improves resistance. Reduce nitrogen.",
        "fungicide": [
            "Metalaxyl + Mancozeb (Ridomil Gold MZ) — 2.5 g/L water — FIRST choice",
            "Cymoxanil 8% + Mancozeb 64% WP — 3 g/L water",
            "Dimethomorph 50 WP — 1 g/L water (curative — use when disease is visible)",
            "Fluopicolide + Propamocarb (Infinito) — 1.6 mL/L water",
        ],
        "cultural": [
            "ACT IMMEDIATELY — spreads within 24–48 hrs in cool wet weather",
            "Destroy ALL infected plants — burn them, do not compost",
            "Avoid waterlogged or poorly drained fields",
            "Use certified disease-free seed tubers only",
            "Hill soil around plants to reduce tuber infection",
        ],
        "organic": "Copper Oxychloride 50 WP — 3 g/L water as preventive. Less curative but safe for organic farming.",
    },

    "Potato|Potato___healthy": {
        "severity": "None — Plant is healthy",
        "cause": "No disease detected",
        "symptoms": "Plant appears healthy with no visible disease symptoms.",
        "fertilizer": "Continue regular NPK schedule. Side-dress urea at tuber initiation. Foliar micronutrient spray if needed.",
        "fungicide": [],
        "cultural": [
            "Maintain regular irrigation without waterlogging",
            "Scout weekly for early signs of disease or pests",
            "Apply preventive fungicide before onset of wet season",
        ],
        "organic": "Preventive neem oil spray 5 mL/L every 2 weeks. Trichoderma soil drench at planting.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # RICE
    # ══════════════════════════════════════════════════════════════════════════
    "Rice|Bacterial Leaf Blight": {
        "severity": "Severe",
        "cause": "Bacteria — Xanthomonas oryzae pv. oryzae",
        "symptoms": "Water-soaked yellowish stripes along leaf margins turning straw-yellow and drying. Wilting of seedlings (Kresek symptom). Bacterial ooze in morning.",
        "fertilizer": "REDUCE nitrogen immediately — excess N worsens BLB drastically. Apply potassium (MOP 60%) at 60 kg/acre.",
        "fungicide": [
            "Copper Oxychloride 50 WP — 3 g/L water (bactericide), spray every 10 days",
            "Bismerthiazol 20 WP — 1.5 g/L water",
            "Streptomycin Sulphate 90 SP + Tetracycline — 0.5 g/L water (use sparingly)",
            "Kasugamycin 3 SL — 2 mL/L water",
        ],
        "cultural": [
            "Drain field for 5–7 days during tillering to reduce spread",
            "Avoid clipping seedling tops during transplanting",
            "Plant resistant varieties — IR64, Swarna Sub1, IR20",
            "Do not use infected fields for seed production",
        ],
        "organic": "Pseudomonas fluorescens bio-agent — 10 g/L water spray every 10 days.",
    },

    "Rice|Brown Spot": {
        "severity": "Moderate — linked to nutrient deficiency",
        "cause": "Fungus — Bipolaris oryzae (Helminthosporium oryzae)",
        "symptoms": "Small circular to oval dark brown spots with grey/whitish center on leaves and grains.",
        "fertilizer": "Apply silicon (calcium silicate 150 kg/acre) — proven to reduce brown spot. Adequate potassium. Zinc sulfate 25 kg/ha if deficient.",
        "fungicide": [
            "Mancozeb 75 WP — 2.5 g/L water, spray at tillering and booting",
            "Iprodione 50 WP — 2 g/L water",
            "Propiconazole 25 EC — 1 mL/L water (highly effective systemic)",
            "Tricyclazole 75 WP — 0.6 g/L water",
        ],
        "cultural": [
            "Fix plant nutrition first — brown spot signals nutrient stress",
            "Treat seeds with Thiram 75 WP 2.5 g/kg seed before sowing",
            "Improve field drainage",
            "Remove infected stubble after harvest",
        ],
        "organic": "Trichoderma viride seed treatment — 4 g/kg seed. Pseudomonas fluorescens foliar spray.",
    },

    "Rice|Healthy Rice Leaf": {
        "severity": "None — Crop is healthy",
        "cause": "No disease detected",
        "symptoms": "Rice crop appears healthy with no visible disease.",
        "fertilizer": "Follow standard NPK schedule for your variety. Apply zinc sulfate 25 kg/ha if yellowing observed.",
        "fungicide": [],
        "cultural": [
            "Maintain proper water management — avoid prolonged flooding",
            "Regular field scouting at least twice a week",
        ],
        "organic": "Pseudomonas fluorescens 10 g/L spray at tillering as preventive bio-fungicide.",
    },

    "Rice|Leaf Blast": {
        "severity": "Severe — can cause 70–80% yield loss",
        "cause": "Fungus — Magnaporthe oryzae (Pyricularia oryzae)",
        "symptoms": "Diamond-shaped lesions with grey/white center and brown border. Neck blast snaps panicle. Node blast blackens nodes.",
        "fertilizer": "STOP all nitrogen immediately if blast is active — N worsens blast severely. Resume only after control. Apply potassium and silicon.",
        "fungicide": [
            "Tricyclazole 75 WP — 0.6 g/L water — MOST EFFECTIVE for blast",
            "Isoprothiolane 40 EC — 1.5 mL/L water (systemic + preventive)",
            "Azoxystrobin 23 SC — 1 mL/L water",
            "Propiconazole 25 EC — 1 mL/L water",
            "Carbendazim 50 WP — 1 g/L water",
        ],
        "cultural": [
            "Spray preventively at tillering and before flowering",
            "Plant resistant varieties — IR36, IR64, Samba Mahsuri",
            "Avoid late application of nitrogen",
            "Maintain optimum plant population — avoid dense planting",
        ],
        "organic": "Pseudomonas fluorescens 10 g/L spray. Good as preventive.",
    },

    "Rice|Leaf scald": {
        "severity": "Moderate",
        "cause": "Fungus — Microdochium oryzae (Rhynchosporium oryzae)",
        "symptoms": "Scalded/bleached appearance on leaf tips and margins. Zig-zag water-soaked lesions turning light tan to grey. Leaf tips die back.",
        "fertilizer": "Balanced NPK. Avoid excess nitrogen. Apply potassium to strengthen leaf tissue.",
        "fungicide": [
            "Propiconazole 25 EC — 1 mL/L water",
            "Tebuconazole 25.9 EC — 1 mL/L water",
            "Mancozeb 75 WP — 2.5 g/L water",
            "Iprodione 50 WP — 2 g/L water",
        ],
        "cultural": [
            "Use disease-free certified seeds",
            "Seed treatment with fungicide before sowing",
            "Crop rotation with non-grass crops",
            "Destroy infected stubble after harvest",
        ],
        "organic": "Trichoderma harzianum 5 g/L foliar spray. Neem seed kernel extract (NSKE) 5%.",
    },

    "Rice|Sheath Blight": {
        "severity": "Moderate to Severe",
        "cause": "Fungus — Rhizoctonia solani",
        "symptoms": "Oval to irregular greenish-grey lesions on leaf sheath near water level. Brown border. White mycelium visible. Lodging in severe cases.",
        "fertilizer": "REDUCE nitrogen — high N dramatically increases severity. Apply in split doses. Maintain adequate potassium and silicon.",
        "fungicide": [
            "Hexaconazole 5 EC — 2 mL/L water — highly effective",
            "Propiconazole 25 EC — 1 mL/L water",
            "Validamycin 3 L — 2 mL/L water (specifically for Rhizoctonia)",
            "Azoxystrobin 23 SC — 1 mL/L water",
            "Carbendazim 50 WP — 1 g/L water",
        ],
        "cultural": [
            "Reduce plant density — wide row spacing reduces canopy humidity",
            "Drain field periodically to reduce humidity at base",
            "Remove infected stubble and burn after harvest",
            "Plant moderately resistant varieties",
        ],
        "organic": "Trichoderma harzianum 10 g/L soil drench + foliar spray. Pseudomonas fluorescens 10 g/L.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CORN / MAIZE
    # ══════════════════════════════════════════════════════════════════════════
    "Corn|Blight": {
        "severity": "Moderate to Severe",
        "cause": "Fungus — Exserohilum turcicum (Northern Leaf Blight)",
        "symptoms": "Long cigar-shaped grayish-green to tan lesions 2.5–15 cm on leaves. Starts on lower leaves and moves up. Entire leaf can die in severe cases.",
        "fertilizer": "Adequate nitrogen in split doses. Potassium improves stalk strength and tolerance.",
        "fungicide": [
            "Propiconazole 25 EC — 1 mL/L water — spray at early tasseling",
            "Azoxystrobin 23 SC — 1 mL/L water",
            "Azoxystrobin + Propiconazole (Amistar Top) — 1 mL/L water (best combination)",
            "Mancozeb 75 WP — 2.5 g/L water (protectant — use preventively)",
            "Tebuconazole 25.9 EC — 1 mL/L water",
        ],
        "cultural": [
            "Plant resistant hybrids — most important management tool",
            "Crop rotation — minimum 1 year away from corn",
            "Deep plow to bury infected crop residue",
            "Avoid dense planting — ensure adequate canopy airflow",
        ],
        "organic": "Trichoderma asperellum bio-fungicide soil drench + foliar spray.",
    },

    "Corn|Healthy": {
        "severity": "None — Crop is healthy",
        "cause": "No disease detected",
        "symptoms": "Corn crop appears healthy with no visible disease symptoms.",
        "fertilizer": "Standard corn NPK schedule — top-dress nitrogen (urea) at knee-high stage. Apply zinc if yellowing between veins.",
        "fungicide": [],
        "cultural": [
            "Maintain regular irrigation",
            "Scout for pests and disease every week",
            "Apply preventive fungicide at tasseling in rainy seasons",
        ],
        "organic": "Preventive Trichoderma viride bio-fungicide soil application at sowing.",
    },

    "Corn|Leaf_Spot": {
        "severity": "Moderate to Severe",
        "cause": "Fungus — Cercospora zeae-maydis (Gray Leaf Spot)",
        "symptoms": "Long narrow tan to gray rectangular lesions parallel to leaf veins. Coalesces and kills entire leaves. Worst in humid conditions with poor airflow.",
        "fertilizer": "Adequate nitrogen and potassium. Potassium improves resistance. Avoid nitrogen deficiency.",
        "fungicide": [
            "Azoxystrobin 23 SC — 1 mL/L water",
            "Propiconazole 25 EC — 1 mL/L water",
            "Pyraclostrobin + Metconazole — 1.5 mL/L water (best for severe infection)",
            "Trifloxystrobin + Tebuconazole — 0.5 mL/L water",
        ],
        "cultural": [
            "Crop rotation with soybean or wheat for at least 1 year",
            "Tillage to bury infected residue below soil surface",
            "Plant resistant hybrid varieties",
            "Spray fungicide at VT (tasseling) growth stage for best protection",
        ],
        "organic": "Neem oil 5 mL/L + copper sulfate 3 g/L combination spray.",
    },

    "Corn|Rust": {
        "severity": "Moderate",
        "cause": "Fungus — Puccinia sorghi (Common Rust)",
        "symptoms": "Small circular to elongated golden-brown to brick-red pustules on both leaf surfaces. Releases powdery rust-colored spores. Severe infection causes yellowing.",
        "fertilizer": "Balanced NPK. Adequate potassium significantly improves rust resistance.",
        "fungicide": [
            "Mancozeb 75 WP — 2.5 g/L water (protectant, use preventively)",
            "Tebuconazole 25.9 EC — 1 mL/L water (systemic, curative)",
            "Trifloxystrobin + Tebuconazole — 0.5 mL/L water",
            "Propiconazole 25 EC — 1 mL/L water",
        ],
        "cultural": [
            "Plant rust-resistant hybrid varieties — most economical solution",
            "Early planting to avoid peak rust season",
            "Monitor weekly during tasseling and silking stages",
        ],
        "organic": "Sulfur 80 WP — 3 g/L water. Highly effective against rust fungi. Spray in cool morning hours.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # SUGARCANE
    # ══════════════════════════════════════════════════════════════════════════
    "Sugarcane|BacterialBlights": {
        "severity": "Severe",
        "cause": "Bacteria — Xanthomonas albilineans",
        "symptoms": "White pencil-line stripes running full length of leaves. Leaf scalding. Wilting and death of entire clumps in severe cases.",
        "fertilizer": "Reduce nitrogen — excess N worsens bacterial spread. Apply potassium at 100 kg K₂O/ha.",
        "fungicide": [
            "No systemic bactericide available — management is primarily cultural",
            "Copper Oxychloride 50 WP — 3 g/L water as protective foliar spray",
            "Streptomycin Sulphate 90 SP — 0.5 g/L water (limited effect)",
        ],
        "cultural": [
            "Use disease-free certified planting material",
            "Hot water treatment of setts at 50°C for 2–3 hours before planting — MOST IMPORTANT",
            "Rogue out and destroy all infected clumps immediately",
            "Disinfect cutting tools with 10% bleach between each cut",
            "Plant resistant varieties — Co-86032, CoM-88121",
        ],
        "organic": "Pseudomonas fluorescens sett treatment — soak in 10 g/L solution for 30 min before planting.",
    },

    "Sugarcane|Healthy": {
        "severity": "None — Crop is healthy",
        "cause": "No disease detected",
        "symptoms": "Sugarcane crop appears healthy.",
        "fertilizer": "Apply NPK at 250:62:112 kg/ha in 3 split doses. Zinc sulfate 25 kg/ha at planting.",
        "fungicide": [],
        "cultural": [
            "Regular trash mulching to conserve moisture",
            "Maintain optimum plant population",
            "Irrigation scheduling based on crop need",
        ],
        "organic": "Trichoderma viride + Pseudomonas fluorescens soil application at planting. Neem cake 250 kg/ha.",
    },

    "Sugarcane|Mosaic": {
        "severity": "Moderate to Severe — no chemical cure",
        "cause": "Virus — Sugarcane Mosaic Virus (SCMV), transmitted by aphids",
        "symptoms": "Alternating light and dark green streaks/mosaic on leaves. Stunted growth, reduced stalk height and sugar content.",
        "fertilizer": "Seaweed extract + micronutrient foliar spray. Balanced NPK to reduce plant stress.",
        "fungicide": [
            "No fungicide or pesticide cures viral diseases",
            "Control aphid vectors immediately to prevent further spread",
        ],
        "cultural": [
            "Control aphid vectors with Imidacloprid 17.8 SL — 0.5 mL/L water",
            "Yellow sticky traps to monitor aphid population",
            "Remove and destroy mosaic-infected plants/stools immediately",
            "Use virus-free certified planting material",
            "Disinfect cutting tools between cuts",
        ],
        "organic": "Neem oil 5 mL/L spray to repel aphid vectors. Yellow sticky traps. Remove infected plants.",
    },

    "Sugarcane|RedRot": {
        "severity": "Severe — one of the most destructive sugarcane diseases",
        "cause": "Fungus — Colletotrichum falcatum",
        "symptoms": "Internal reddening of stalk tissue with white patches. Leaves wither and dry. Sour/alcoholic smell from infected stalk. Stalk shrivels.",
        "fertilizer": "Adequate potassium (MOP 150 kg/ha). Avoid waterlogging which promotes fungal spread.",
        "fungicide": [
            "Carbendazim 50 WP — sett treatment: 1 g/L water, soak 30 min — ESSENTIAL",
            "Tridemorph 80 EC — 1 mL/L water foliar spray",
            "Propiconazole 25 EC — 1 mL/L water foliar spray",
            "Thiophanate-methyl 70 WP — 1 g/L water",
        ],
        "cultural": [
            "Use resistant varieties — most critical: Co-0238, CoM-0265",
            "Hot water sett treatment: 50°C for 2 hours before planting",
            "Avoid waterlogging — improve field drainage urgently",
            "Do NOT retain ratoon from infected crop",
            "Destroy infected stools by burning",
        ],
        "organic": "Trichoderma harzianum sett treatment — soak in 10 g/L water for 1 hour before planting.",
    },

    "Sugarcane|Rust": {
        "severity": "Moderate",
        "cause": "Fungus — Puccinia melanocephala (brown rust) / Puccinia kuehnii (orange rust)",
        "symptoms": "Small elongated orange-brown pustules on leaf surfaces. Powdery rust-colored spore masses. Heavy infection causes premature leaf drying.",
        "fertilizer": "Balanced nutrition. Potassium (MOP 100 kg/ha) improves rust tolerance.",
        "fungicide": [
            "Propiconazole 25 EC — 1 mL/L water — spray at first pustule appearance",
            "Tebuconazole 25.9 EC — 1 mL/L water",
            "Azoxystrobin 23 SC — 1 mL/L water",
            "Trifloxystrobin 25 WG — 0.5 g/L water",
        ],
        "cultural": [
            "Plant rust-resistant varieties where available",
            "Early planting to escape peak rust season",
            "Avoid high plant density — maintain airflow",
            "Scout weekly for early pustule detection",
        ],
        "organic": "Sulfur 80 WP — 3 g/L water spray every 10 days during rust season.",
    },

    "Sugarcane|Yellow": {
        "severity": "Moderate to Severe",
        "cause": "Phytoplasma / Sugarcane Yellow Leaf Virus (SCYLV), transmitted by aphids",
        "symptoms": "Yellowing of midrib on top leaves (first sign). Spreads to entire leaf. Stunted growth, reduced tillering and sugar content.",
        "fertilizer": "Zinc, magnesium, manganese micronutrient spray. Foliar 2% urea + micronutrients. Avoid over-applying nitrogen.",
        "fungicide": [
            "No fungicide cures phytoplasma/viral diseases",
            "Control aphid vectors immediately",
        ],
        "cultural": [
            "Control aphids with Thiamethoxam 25 WG — 0.3 g/L water",
            "Imidacloprid 17.8 SL — 0.5 mL/L water foliar spray",
            "Yellow sticky traps (30/acre) to monitor aphids",
            "Use disease-free planting material from certified nurseries",
            "Remove and destroy yellow leaf affected stools",
        ],
        "organic": "Neem oil 5 mL/L spray weekly to repel aphid vectors. Yellow sticky traps.",
    },

    # ══════════════════════════════════════════════════════════════════════════
    # TOMATO
    # ══════════════════════════════════════════════════════════════════════════
    "Tomato|Tomato___Bacterial_spot": {
        "severity": "Moderate to Severe",
        "cause": "Bacteria — Xanthomonas campestris pv. vesicatoria",
        "symptoms": "Small water-soaked spots on leaves turning dark brown with yellow halo. Raised scabby spots on fruit. Severe defoliation.",
        "fertilizer": "Reduce nitrogen. Apply calcium nitrate (15.5-0-0 + 19% Ca) to strengthen cell walls.",
        "fungicide": [
            "Copper Oxychloride 50 WP — 3 g/L water, spray every 7 days",
            "Copper Hydroxide 77 WP — 3 g/L water",
            "Streptomycin Sulphate 90 SP — 0.5 g/L water (use sparingly)",
            "Kasugamycin 3 SL — 2 mL/L water",
        ],
        "cultural": [
            "Use certified disease-free seeds — hot water treatment (50°C for 25 min)",
            "Avoid working in wet fields — bacteria spread via water splash",
            "Remove and burn infected plant debris",
            "Avoid overhead irrigation — drip irrigation preferred",
        ],
        "organic": "Copper-based bactericide + neem extract spray weekly. Pseudomonas fluorescens 10 g/L.",
    },

    "Tomato|Tomato___Early_blight": {
        "severity": "Moderate",
        "cause": "Fungus — Alternaria solani",
        "symptoms": "Concentric ring spots on lower/older leaves. Dark brown lesions with yellow halo. Premature defoliation from base upward.",
        "fertilizer": "NPK 12-32-16 at flowering. Adequate phosphorus for root strength. Calcium nitrate foliar spray.",
        "fungicide": [
            "Mancozeb 75 WP — 2.5 g/L water, spray every 7 days",
            "Iprodione 50 WP — 2 g/L water",
            "Tebuconazole 25.9 EC — 1 mL/L water (systemic)",
            "Azoxystrobin 23 SC — 1 mL/L water",
        ],
        "cultural": [
            "Remove and destroy infected lower leaves promptly",
            "Mulch around plants to prevent soil splash",
            "Water in morning so leaves dry during daytime",
            "Stake/trellis plants for airflow",
        ],
        "organic": "Neem oil 5 mL/L + copper sulfate 3 g/L spray every 7 days.",
    },

    "Tomato|Tomato___Late_blight": {
        "severity": "Severe — can destroy crop in 7–10 days",
        "cause": "Water mold — Phytophthora infestans",
        "symptoms": "Large irregular greasy dark lesions on leaves and stems. White sporulation on underside in humid conditions. Brown sunken lesions on fruit.",
        "fertilizer": "Apply potassium sulfate (SOP 0-0-50). Reduce nitrogen immediately.",
        "fungicide": [
            "Metalaxyl + Mancozeb (Ridomil Gold) — 2.5 g/L water — spray immediately",
            "Cymoxanil + Famoxadone — 0.6 g/L water",
            "Fluopicolide + Propamocarb (Infinito) — 1.6 mL/L water",
            "Dimethomorph 50 WP — 1 g/L water (curative)",
        ],
        "cultural": [
            "IMMEDIATE ACTION — disease spreads in 24–48 hrs",
            "Destroy and burn infected plants — do not compost",
            "Stop overhead irrigation completely",
            "Improve field drainage urgently",
        ],
        "organic": "Copper Hydroxide 77 WP — 3 g/L water as preventive. Limited curative effect.",
    },

    "Tomato|Tomato___Leaf_Mold": {
        "severity": "Moderate",
        "cause": "Fungus — Passalora fulva (Fulvia fulva)",
        "symptoms": "Pale green/yellow patches on upper leaf surface. Olive-green to brown velvety mold on underside. Infected leaves curl, wither and drop.",
        "fertilizer": "Balanced fertilization — avoid excess nitrogen which promotes dense canopy and humidity.",
        "fungicide": [
            "Chlorothalonil 75 WP — 2 g/L water",
            "Mancozeb + Carbendazim — 2 g/L water",
            "Difenoconazole 25 EC — 0.5 mL/L water (systemic)",
        ],
        "cultural": [
            "Improve ventilation — leaf mold thrives in high humidity",
            "Maintain relative humidity below 85%",
            "Remove and destroy infected leaves promptly",
            "Avoid overhead watering",
        ],
        "organic": "Trichoderma viride bio-fungicide — 5 g/L water spray. Neem oil 5 mL/L.",
    },

    "Tomato|Tomato___Septoria_leaf_spot": {
        "severity": "Moderate",
        "cause": "Fungus — Septoria lycopersici",
        "symptoms": "Many small circular spots with dark brown border and light grey/white center. Tiny black specks inside spots. Rapid defoliation from base upward.",
        "fertilizer": "Adequate calcium and potassium. Foliar calcium spray reduces infection entry points.",
        "fungicide": [
            "Mancozeb 75 WP — 2.5 g/L water, spray every 7 days",
            "Copper Oxychloride 50 WP — 3 g/L water",
            "Azoxystrobin 23 SC — 1 mL/L water",
            "Chlorothalonil 75 WP — 2 g/L water",
        ],
        "cultural": [
            "Remove and destroy infected lower leaves immediately",
            "Avoid overhead watering — water at base",
            "Crop rotation — minimum 2 years without tomato",
            "Mulch around base to prevent soil splash",
        ],
        "organic": "Neem oil 5 mL/L spray every 7 days. Copper-based spray as backup.",
    },

    "Tomato|Tomato___Spider_mites": {
        "severity": "Moderate to Severe — pest, not a fungus",
        "cause": "Pest — Tetranychus urticae (Two-spotted spider mite)",
        "symptoms": "Fine stippling/speckling on leaf surface. Webbing on underside of leaves. Leaves turn bronze/yellow and dry out.",
        "fertilizer": "Adequate watering reduces plant stress — drought-stressed plants attract mites.",
        "fungicide": [
            "Abamectin 1.8 EC (miticide/acaricide) — 1 mL/L water — BEST CHOICE",
            "Spiromesifen 22.9 SC — 1 mL/L water",
            "Bifenazate 43 SC — 1.5 mL/L water",
            "Rotate miticides — mites develop resistance quickly",
        ],
        "cultural": [
            "Spray plants forcefully with water to dislodge mites from undersides",
            "Release predatory mites (Phytoseiulus persimilis) for biological control",
            "Avoid dusty conditions — mites thrive in dust",
            "Remove heavily infested leaves",
        ],
        "organic": "Neem oil 10 mL/L + liquid soap 5 mL/L spray. Garlic extract spray. Sulfur 80 WP 3 g/L.",
    },

    "Tomato|Tomato___Target_Spot": {
        "severity": "Moderate",
        "cause": "Fungus — Corynespora cassiicola",
        "symptoms": "Brown spots with concentric rings on leaves, stems and fruit. Can affect all above-ground parts. Premature defoliation.",
        "fertilizer": "Balanced NPK. Avoid excess nitrogen. Calcium nitrate foliar spray.",
        "fungicide": [
            "Azoxystrobin + Difenoconazole — 1 mL/L water — BEST combination",
            "Boscalid + Pyraclostrobin — 0.8 g/L water",
            "Tebuconazole 25.9 EC — 1 mL/L water",
            "Mancozeb 75 WP — 2.5 g/L water (protectant)",
        ],
        "cultural": [
            "Crop rotation — at least 2 years",
            "Remove all plant debris after harvest — burn it",
            "Improve airflow by pruning lower leaves and staking",
        ],
        "organic": "Trichoderma harzianum bio-fungicide application to soil and foliage.",
    },

    "Tomato|Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "severity": "Severe — no chemical cure",
        "cause": "Virus — TYLCV, spread exclusively by whitefly (Bemisia tabaci)",
        "symptoms": "Upward curling of leaves. Yellowing of leaf margins. Stunted plant. Severely reduced fruit set. Young leaves appear crumpled.",
        "fertilizer": "Micronutrient spray — zinc + boron + molybdenum. Balanced NPK to reduce stress.",
        "fungicide": [
            "No fungicide or pesticide can cure viral diseases",
            "Control whitefly vectors IMMEDIATELY and aggressively",
        ],
        "cultural": [
            "Control whitefly with Thiamethoxam 25 WG — 0.3 g/L water",
            "Imidacloprid 17.8 SL — 0.5 mL/L water as soil drench at transplanting",
            "Yellow sticky traps — 20–30 per acre",
            "Reflective silver mulch strongly repels whiteflies",
            "Remove and destroy infected plants immediately",
            "Use insect-proof nets (50 mesh) in nursery",
        ],
        "organic": "Neem oil 5 mL/L spray to repel whiteflies. Yellow sticky traps. Remove infected plants.",
    },

    "Tomato|Tomato___Tomato_mosaic_virus": {
        "severity": "Severe — no chemical cure",
        "cause": "Virus — Tomato Mosaic Virus (ToMV), highly contagious, spreads by contact",
        "symptoms": "Mosaic pattern of light and dark green on leaves. Distorted/curled leaves. Stunted growth. Internal browning of fruit.",
        "fertilizer": "Seaweed extract + micronutrient foliar spray to boost plant immunity.",
        "fungicide": [
            "No fungicide effective against viruses",
            "Remove and destroy infected plants immediately",
        ],
        "cultural": [
            "Use certified virus-free seeds — hot water treatment (50°C for 25 min)",
            "Control aphid vectors with Imidacloprid 17.8 SL — 0.5 mL/L water",
            "Wash hands with soap before handling plants",
            "Disinfect all tools with 10% bleach between plants",
            "AVOID TOBACCO near plants — TMV transfers from tobacco via hands",
            "Plant ToMV-resistant varieties",
        ],
        "organic": "Neem oil spray to control aphid vectors. Skim milk spray (10%) reported to reduce spread.",
    },

    "Tomato|Tomato___healthy": {
        "severity": "None — Plant is healthy",
        "cause": "No disease detected",
        "symptoms": "Tomato plant appears healthy with no visible disease symptoms.",
        "fertilizer": "Continue regular fertigation: NPK + calcium nitrate + magnesium sulfate. Foliar micronutrient spray monthly.",
        "fungicide": [],
        "cultural": [
            "Regular scouting — check leaves top and bottom twice a week",
            "Maintain proper drip irrigation and drainage",
            "Preventive copper spray before rainy season",
        ],
        "organic": "Preventive neem oil 5 mL/L spray every 10–14 days. Trichoderma soil drench monthly.",
    },
}


def get_treatment(plant: str, raw_class: str) -> Optional[dict]:
    """Look up treatment by plant name and exact disease folder name."""
    key = f"{plant}|{raw_class}"
    return TREATMENTS.get(key)