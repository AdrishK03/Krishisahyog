# ─────────────────────────────────────────────────────────────────────────────
# disease_classes.py
# EXACT subfolder names from each plant's disease dataset
# Order MUST match the order your Keras model was trained on
# ─────────────────────────────────────────────────────────────────────────────

# Maps each plant → list of disease class folder names (in training order)
DISEASE_CLASSES: dict[str, list[str]] = {
    "Potato": [
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
    ],
    "Rice": [
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Leaf scald",
        "Sheath Blight",
    ],
    "Corn": [
        "Blight",
        "Healthy",
        "Leaf_Spot",
        "Rust",
    ],
    "Sugarcane": [
        "BacterialBlights",
        "Healthy",
        "Mosaic",
        "RedRot",
        "Rust",
        "Yellow",
    ],
    "Tomato": [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ],
}

# Clean display labels shown in the UI (raw folder name → readable name)
DISEASE_DISPLAY: dict[str, str] = {
    # Potato
    "Potato___Early_blight":                   "Early Blight",
    "Potato___Late_blight":                    "Late Blight",
    "Potato___healthy":                        "Healthy",
    # Rice
    "Bacterial Leaf Blight":                   "Bacterial Leaf Blight",
    "Brown Spot":                              "Brown Spot",
    "Healthy Rice Leaf":                       "Healthy",
    "Leaf Blast":                              "Leaf Blast",
    "Leaf scald":                              "Leaf Scald",
    "Sheath Blight":                           "Sheath Blight",
    # Corn
    "Blight":                                  "Blight",
    "Healthy":                                 "Healthy",
    "Leaf_Spot":                               "Leaf Spot",
    "Rust":                                    "Rust",
    # Sugarcane
    "BacterialBlights":                        "Bacterial Blight",
    "Mosaic":                                  "Mosaic Disease",
    "RedRot":                                  "Red Rot",
    "Yellow":                                  "Yellow Leaf Disease",
    # Tomato
    "Tomato___Bacterial_spot":                 "Bacterial Spot",
    "Tomato___Early_blight":                   "Early Blight",
    "Tomato___Late_blight":                    "Late Blight",
    "Tomato___Leaf_Mold":                      "Leaf Mold",
    "Tomato___Septoria_leaf_spot":             "Septoria Leaf Spot",
    "Tomato___Spider_mites":                   "Spider Mites",
    "Tomato___Target_Spot":                    "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":  "Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus":            "Tomato Mosaic Virus",
    "Tomato___healthy":                        "Healthy",
}