# generate_dic_maps.py
import numpy as np
import muDIC as md

def run_synthetic_dic_pipeline():
    print("====================================================")
    print("       INITIALIZING µDIC MECHANICS ENGINE (STEP 3)  ")
    print("====================================================")
    
    # 1. Establish custom spatial coordinate geometry for the scaffold layer
    grid_size = 300
    x = np.linspace(0, 15, grid_size)
    y = np.linspace(0, 15, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 2. Build an analytical high-frequency speckle map
    print("[+] Compiling analytical tracking speckle matrix...")
    speckle_texture = (
        np.sin(X * 5.0) * np.cos(Y * 5.0) + 
        np.sin(X * 2.2) * np.sin(Y * 1.8) + 
        np.cos(X * 8.5 + Y * 4.1)
    )
    speckle_texture = ((speckle_texture - speckle_texture.min()) / 
                       (speckle_texture.max() - speckle_texture.min()) * 255.0)
    image_reference = speckle_texture.astype(np.float32)
    
    # 3. Apply a structural deformation field (Simulating localized scaffold warping defect)
    print("[+] Simulating mechanical displacement defect field...")
    X_def = X + 0.12 * np.sin(X / 2.5)
    Y_def = Y + 0.04 * np.cos(X / 3.0)
    
    deformed_texture = (
        np.sin(X_def * 5.0) * np.cos(Y_def * 5.0) + 
        np.sin(X_def * 2.2) * np.sin(Y_def * 1.8) + 
        np.cos(X_def * 8.5 + Y_def * 4.1)
    )
    deformed_texture = ((deformed_texture - deformed_texture.min()) / 
                        (deformed_texture.max() - deformed_texture.min()) * 255.0)
    image_deformed = deformed_texture.astype(np.uint8) # solver handles tracking comfortably via unsigned inputs
    
    # 4. Wrap images in official muDIC ImageStack container
    print("[+] Instantiating official µDIC ImageStack instance...")
    image_stack = md.IO.image_stack_from_list([image_reference, image_deformed])
    
    # 5. Construct the Discretization Mesh matching the exact inspected signature
    print("[+] Generating background automated structured mesh grid...")
    q4_element = md.mesh.meshUtilities.Q4()
    mesh = md.mesh.meshUtilities.Mesh(
        q4_element,
        5,    # corner1_x
        295,  # corner2_x
        5,    # corner1_y
        295,  # corner2_y
        10,   # n_elx
        10    # n_ely
    )
    
    # 6. Execute Solver with corrected parameter order using explicit keywords
    print("[+] Launching µDIC optical flow correlation solver...")
    dic_inputs = md.DICInput(mesh=mesh, image_stack=image_stack)
    dic_analysis = md.DICAnalysis(dic_inputs)
    dic_results = dic_analysis.run()
    
    # 7. Extract full-field continuum strain fields using the confirmed Fields class
    print("[+] Extracting continuum mechanics true strain fields...")
    fields = md.post.Fields(dic_results)
    strain_tensor = fields.true_strain()
    
    # Isolate the peak longitudinal strain value to act as our scalar tracking metric
    max_strain_val = float(np.max(np.abs(strain_tensor)))
    
    print(f"[+] Extraction complete. Tensor Matrix Data Shape: {strain_tensor.shape}")
    print(f"[+] Peak true strain tensor amplitude calculated: {max_strain_val:.6f}")
    print("====================================================")
    
    return max_strain_val

if __name__ == "__main__":
    peak_strain = run_synthetic_dic_pipeline()