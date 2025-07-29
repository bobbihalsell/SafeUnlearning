import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset

def analyze_attack_results(results):
    """
    Analyze and display attack results.
    This is the main function called from main.py
    """
    print('\n' + '='*60)
    print('ATTACK RESULTS ANALYSIS')
    print('='*60)
    
    # Check if attack was successful
    if 'forget_comparison' in results:
        forget_comp = results['forget_comparison']
        
        print("\nEstimated vs True Forget Subspace Similarity:")
        print("-" * 50)
        
        similarities = []
        for layer_name, comp in forget_comp.items():
            if comp:
                sim = comp['mean_cosine']
                similarities.append(sim)
                print(f"{layer_name:25s}: cosine={sim:.3f}, angle={comp['mean_angle_degrees']:5.1f}°")
        
        if similarities:
            mean_similarity = np.mean(similarities)
            print(f"\nOverall Mean Cosine Similarity: {mean_similarity:.3f}")
            
            if mean_similarity > 0.8:
                print("🎯 ATTACK SUCCESSFUL: High similarity to true forget subspace!")
            elif mean_similarity > 0.6:
                print("⚠️  ATTACK PARTIALLY SUCCESSFUL: Moderate similarity")
            else:
                print("❌ ATTACK FAILED: Low similarity to true forget subspace")
    
    if 'retain_comparison' in results:
        retain_comp = results['retain_comparison']
        
        print("\nEstimated Forget vs True Retain Subspace (should be different):")
        print("-" * 50)
        
        retain_similarities = []
        for layer_name, comp in retain_comp.items():
            if comp:
                sim = comp['mean_cosine']
                retain_similarities.append(sim)
                print(f"{layer_name:25s}: cosine={sim:.3f}, angle={comp['mean_angle_degrees']:5.1f}°")
        
        if retain_similarities:
            mean_retain_sim = np.mean(retain_similarities)
            print(f"\nMean Cosine with Retain: {mean_retain_sim:.3f}")
            
            if mean_retain_sim < 0.3:
                print("✅ GOOD: Low similarity to retain subspace (as expected)")
            else:
                print("⚠️  WARNING: High similarity to retain subspace")
    
    # Test gradient projection effectiveness
    print("\n" + "="*60)
    print("GRADIENT PROJECTION TEST")
    print("="*60)
    
    if 'original_gradients' in results and 'projected_gradients' in results:
        original_grads = results['original_gradients']
        projected_grads = results['projected_gradients']
        
        # Calculate how much the gradients changed
        total_original_norm = 0
        total_projection_norm = 0
        
        for param_name in original_grads:
            if param_name in projected_grads:
                orig_grad = original_grads[param_name]
                proj_grad = projected_grads[param_name]
                
                if orig_grad is not None and proj_grad is not None:
                    diff = orig_grad - proj_grad
                    
                    orig_norm = torch.norm(orig_grad).item()
                    diff_norm = torch.norm(diff).item()
                    
                    total_original_norm += orig_norm
                    total_projection_norm += diff_norm
                    
                    if '.weight' in param_name:
                        layer_name = param_name.replace('.weight', '')
                        if 'projection_matrices' in results and layer_name in results['projection_matrices']:
                            relative_change = diff_norm / (orig_norm + 1e-8)
                            print(f"{layer_name:25s}: projection strength = {relative_change:.3f}")
        
        overall_projection_strength = total_projection_norm / (total_original_norm + 1e-8)
        print(f"\nOverall Projection Strength: {overall_projection_strength:.3f}")

def alternative_attack_metrics(original_model, unlearned_model, background_data,
                              forget_data, retain_data, device='cuda'):
    """
    Test alternative metrics beyond subspace cosine similarity.
    """
    
    print("🔍 ALTERNATIVE ATTACK METRICS")
    print("="*60)
    
    from recover import CachedGradientRecoveryPipeline
    pipeline = CachedGradientRecoveryPipeline(device=device, verbose=False)
    
    # ==================================================================
    # METRIC 1: Gradient Magnitude Analysis
    # ==================================================================
    
    print("\n1. 📊 GRADIENT MAGNITUDE ANALYSIS")
    print("-" * 40)
    
    # Get gradients from both models on same data
    orig_grads = pipeline.get_gradients(original_model, background_data, max_batches=5)
    unl_grads = pipeline.get_gradients(unlearned_model, background_data, max_batches=5)
    
    # Calculate gradient differences
    grad_differences = {}
    for param_name in orig_grads:
        if param_name in unl_grads and orig_grads[param_name] is not None:
            grad_differences[param_name] = orig_grads[param_name] - unl_grads[param_name]
    
    # Get true forget/retain gradients
    forget_grads = pipeline.get_gradients(original_model, forget_data, max_batches=3)
    retain_grads = pipeline.get_gradients(original_model, retain_data, max_batches=3)
    
    # Compare gradient magnitudes (not just directions)
    print("Gradient magnitude comparison:")
    promising_layers = []
    
    for param_name in grad_differences:
        if '.weight' in param_name and param_name in forget_grads and param_name in retain_grads:
            diff_mag = torch.norm(grad_differences[param_name]).item()
            forget_mag = torch.norm(forget_grads[param_name]).item()
            retain_mag = torch.norm(retain_grads[param_name]).item()
            
            forget_ratio = diff_mag / (forget_mag + 1e-8)
            retain_ratio = diff_mag / (retain_mag + 1e-8)
            
            layer_name = param_name.replace('.weight', '')
            print(f"  {layer_name:25s}: diff/forget={forget_ratio:.3f}, diff/retain={retain_ratio:.3f}")
            
            if forget_ratio > 2 * retain_ratio:
                print(f"    ✅ Layer {layer_name} shows forget signal!")
                promising_layers.append(layer_name)
    
    # ==================================================================
    # METRIC 2: Parameter Change Analysis
    # ==================================================================
    
    print("\n2. 📊 PARAMETER CHANGE ANALYSIS")
    print("-" * 40)
    
    # Direct parameter differences between models
    param_changes = {}
    for (name1, p1), (name2, p2) in zip(original_model.named_parameters(), 
                                        unlearned_model.named_parameters()):
        param_changes[name1] = p1 - p2
    
    # Analyze which layers changed most
    layer_changes = []
    for name, change in param_changes.items():
        if '.weight' in name:
            change_norm = torch.norm(change).item()
            # Get parameter norm properly
            param_norm = torch.norm(dict(original_model.named_parameters())[name]).item()
            relative_change = change_norm / (param_norm + 1e-8)
            
            layer_name = name.replace('.weight', '')
            layer_changes.append((layer_name, relative_change))
    
    # Sort by change magnitude
    layer_changes.sort(key=lambda x: x[1], reverse=True)
    
    print("Layers with most parameter changes:")
    for layer_name, change in layer_changes[:10]:
        print(f"  {layer_name:25s}: relative_change = {change:.6f}")
    
    # Focus attack on most-changed layers
    most_changed_layers = [layer for layer, _ in layer_changes[:5]]
    print(f"\n💡 Focus attack on these layers: {most_changed_layers}")
    
    # ==================================================================
    # METRIC 3: Loss Landscape Analysis
    # ==================================================================
    
    print("\n3. 📊 LOSS LANDSCAPE ANALYSIS")
    print("-" * 40)
    
    loss_analysis = analyze_loss_landscapes(
        original_model, unlearned_model, forget_data, retain_data, device
    )
    
    return {
        'gradient_magnitude': True,
        'parameter_changes': layer_changes,
        'loss_landscape': loss_analysis,
        'most_changed_layers': most_changed_layers,
        'promising_layers': promising_layers
    }

def analyze_loss_landscapes(original_model, unlearned_model, forget_data, retain_data, device):
    """
    Analyze how loss landscapes differ for forget vs retain data.
    """
    
    criterion = nn.CrossEntropyLoss()
    
    # Calculate losses on forget data
    forget_loss_orig = calculate_loss(original_model, forget_data, criterion, device)
    forget_loss_unl = calculate_loss(unlearned_model, forget_data, criterion, device)
    
    # Calculate losses on retain data  
    retain_loss_orig = calculate_loss(original_model, retain_data, criterion, device)
    retain_loss_unl = calculate_loss(unlearned_model, retain_data, criterion, device)
    
    forget_loss_change = forget_loss_unl - forget_loss_orig
    retain_loss_change = retain_loss_unl - retain_loss_orig
    
    print(f"Loss changes:")
    print(f"  Forget: {forget_loss_orig:.3f} → {forget_loss_unl:.3f} (Δ: {forget_loss_change:+.3f})")
    print(f"  Retain: {retain_loss_orig:.3f} → {retain_loss_unl:.3f} (Δ: {retain_loss_change:+.3f})")
    
    if forget_loss_change > 2 * abs(retain_loss_change):
        print("  ✅ Clear forget signal: forget loss increased much more than retain")
        return True
    else:
        print("  ❌ Unclear signal: similar loss changes for forget and retain")
        return False

def calculate_loss(model, data_loader, criterion, device, max_batches=5):
    """Helper to calculate average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, (data, targets) in enumerate(data_loader):
            if i >= max_batches:
                break
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

# ==================================================================
# IDEAL TEST SCENARIOS WHERE ATTACK SHOULD DEFINITELY WORK
# ==================================================================

def test_attack_on_ideal_scenarios(original_model, device='cuda'):
    """
    Test the gradient recovery attack on all ideal scenarios.
    """
    
    print("🧪 TESTING ATTACK ON IDEAL SCENARIOS")
    print("="*60)
    
    # Create all scenarios
    scenarios = create_ideal_test_scenarios(original_model, device)
    
    results = {}
    
    for scenario_name, scenario_data in scenarios:
        print(f"\n🎯 TESTING: {scenario_name}")
        print("-" * 50)
        
        # Run attack on this scenario
        success = test_single_scenario(scenario_data, scenario_name, device)
        results[scenario_name] = success
        
        if success:
            print(f"✅ {scenario_name}: ATTACK SUCCESSFUL!")
        else:
            print(f"❌ {scenario_name}: Attack failed")
    
    # Summary
    print(f"\n📊 SCENARIO TEST RESULTS:")
    print("-" * 30)
    for scenario_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {scenario_name:25s}: {status}")
    
    successful_scenarios = sum(results.values())
    total_scenarios = len(results)
    
    print(f"\nOverall: {successful_scenarios}/{total_scenarios} scenarios successful")
    
    if successful_scenarios == 0:
        print("❌ CRITICAL: Attack failed on ALL ideal scenarios!")
        print("   This suggests fundamental issues with the attack implementation.")
    elif successful_scenarios < total_scenarios:
        print("⚠️  Attack works on some scenarios but not others.")
        print("   This suggests the method is correct but limited in scope.")
    else:
        print("✅ SUCCESS: Attack works on all ideal scenarios!")
        print("   The method is correct - real data may just be too similar.")
    
    return results

def create_ideal_test_scenarios(original_model, device='cuda'):
    """
    Create synthetic scenarios where the gradient recovery attack should definitely work.
    """
    
    print("\n🧪 CREATING IDEAL TEST SCENARIOS")
    print("="*60)
    
    scenarios = []
    
    # ==================================================================
    # SCENARIO 1: Complete Class Removal
    # ==================================================================
    
    print("\n1. 🎯 SCENARIO: Complete Class Removal")
    print("-" * 40)
    
    scenario1 = create_class_removal_scenario(original_model, device)
    scenarios.append(("Complete Class Removal", scenario1))
    
    # ==================================================================
    # SCENARIO 2: Small Coherent Subset
    # ==================================================================
    
    print("\n2. 🎯 SCENARIO: Small Coherent Subset")
    print("-" * 40)
    
    scenario2 = create_small_subset_scenario(original_model, device)
    scenarios.append(("Small Subset", scenario2))
    
    return scenarios

def create_class_removal_scenario(original_model, device):
    """
    Create a scenario where entire classes are removed.
    This should be the easiest case for gradient recovery.
    """
    
    print("Creating complete class removal scenario...")
    
    # Example: CIFAR-10 with classes 0,1,2 forgotten, classes 3,4,5,6,7,8,9 retained
    num_classes = 10
    input_size = (3, 32, 32)
    
    # Create synthetic data with clear class separation
    forget_classes = [0, 1, 2]  # Cars, trucks, ships
    retain_classes = [3, 4, 5, 6, 7, 8, 9]  # Everything else
    
    # Generate synthetic data
    forget_data = generate_class_data(forget_classes, input_size, device, samples_per_class=100)
    retain_data = generate_class_data(retain_classes, input_size, device, samples_per_class=100)
    background_data = generate_class_data(list(range(num_classes)), input_size, device, samples_per_class=50)
    
    # Create "unlearned" model by fine-tuning only on retain data
    unlearned_model = create_class_unlearned_model(original_model, retain_data, device)
    
    print(f"  ✅ Created class removal scenario:")
    print(f"     Forget classes: {forget_classes}")
    print(f"     Retain classes: {retain_classes}")
    print(f"     This should show CLEAR subspace differences!")
    
    return {
        'original_model': original_model,
        'unlearned_model': unlearned_model,
        'background_data': background_data,
        'forget_data': forget_data,
        'retain_data': retain_data,
        'description': f"Complete removal of classes {forget_classes}"
    }

def create_small_subset_scenario(original_model, device):
    """
    Create scenario with very small, coherent forget set.
    """
    
    print("Creating small coherent subset scenario...")
    
    input_size = (3, 32, 32)
    
    # Very small forget set (50 samples) with very specific pattern
    forget_data = generate_specific_pattern_data("red_squares", input_size, device, num_samples=50)
    retain_data = generate_specific_pattern_data("blue_circles", input_size, device, num_samples=1000)
    background_data = generate_specific_pattern_data("mixed_patterns", input_size, device, num_samples=300)
    
    # Create unlearned model with targeted forgetting
    unlearned_model = create_targeted_unlearned_model(original_model, forget_data, retain_data, device)
    
    print(f"  ✅ Created small subset scenario:")
    print(f"     Forget: 50 red squares (very specific)")
    print(f"     Retain: 1000 blue circles (general)")
    print(f"     This should show TARGETED subspace removal!")
    
    return {
        'original_model': original_model,
        'unlearned_model': unlearned_model,
        'background_data': background_data,
        'forget_data': forget_data,
        'retain_data': retain_data,
        'description': "Small, highly specific forget set"
    }

# ==================================================================
# HELPER FUNCTIONS FOR SYNTHETIC DATA GENERATION
# ==================================================================

def generate_class_data(classes, input_size, device, samples_per_class=100):
    """Generate synthetic data for specific classes."""
    
    all_data = []
    all_labels = []
    
    for class_id in classes:
        # Generate class-specific data (simplified)
        class_data = torch.randn(samples_per_class, *input_size, device=device)
        # Add class-specific bias to make classes distinguishable
        class_data += torch.tensor([class_id * 0.1, 0, 0], device=device).view(1, 3, 1, 1)
        
        class_labels = torch.full((samples_per_class,), class_id, device=device)
        
        all_data.append(class_data)
        all_labels.append(class_labels)
    
    combined_data = torch.cat(all_data, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)
    
    dataset = TensorDataset(combined_data, combined_labels)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def generate_specific_pattern_data(pattern_type, input_size, device, num_samples=500):
    """Generate data with very specific patterns."""
    
    data = torch.randn(num_samples, *input_size, device=device) * 0.1  # Low noise base
    labels = torch.randint(0, 10, (num_samples,), device=device)
    
    if pattern_type == "red_squares":
        # Add red squares in center
        center = input_size[1] // 2
        data[:, 0, center-4:center+4, center-4:center+4] = 1.0  # Red channel
    elif pattern_type == "blue_circles":
        # Add blue circles (simplified as squares for now)
        center = input_size[1] // 2
        data[:, 2, center-6:center+6, center-6:center+6] = 1.0  # Blue channel
    # "mixed_patterns" uses the random base
    
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=32, shuffle=True)

# ==================================================================
# HELPER FUNCTIONS FOR CREATING "UNLEARNED" MODELS
# ==================================================================

def create_class_unlearned_model(original_model, retain_data, device):
    """Create unlearned model by fine-tuning on retain data only."""
    
    import copy
    unlearned_model = copy.deepcopy(original_model)
    unlearned_model.train()
    
    optimizer = torch.optim.SGD(unlearned_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("  Fine-tuning model on retain data only...")
    
    for epoch in range(3):  # Quick fine-tuning
        for batch_idx, (data, targets) in enumerate(retain_data):
            if batch_idx >= 10:  # Limit batches
                break
                
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = unlearned_model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    print("  ✅ Created class-unlearned model")
    return unlearned_model

def create_targeted_unlearned_model(original_model, forget_data, retain_data, device):
    """Create unlearned model with targeted forgetting."""
    
    import copy
    unlearned_model = copy.deepcopy(original_model)
    unlearned_model.train()
    
    optimizer = torch.optim.SGD(unlearned_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("  Applying targeted unlearning...")
    
    # Gradient ascent on forget data (to increase loss)
    for epoch in range(2):
        for batch_idx, (data, targets) in enumerate(forget_data):
            if batch_idx >= 5:
                break
                
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = unlearned_model(data)
            loss = criterion(outputs, targets)
            (-loss).backward()  # Gradient ascent!
            optimizer.step()
    
    # Fine-tune on retain data
    for epoch in range(1):
        for batch_idx, (data, targets) in enumerate(retain_data):
            if batch_idx >= 5:
                break
                
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = unlearned_model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    print("  ✅ Created targeted-unlearned model")
    return unlearned_model

def test_single_scenario(scenario_data, scenario_name, device):
    """Test attack on a single scenario."""
    
    from recover import run_cached_gradient_recovery_attack
    
    try:
        # Run the attack
        results, pipeline = run_cached_gradient_recovery_attack(
            original_model=scenario_data['original_model'],
            unlearned_model=scenario_data['unlearned_model'],
            background_data=scenario_data['background_data'],
            forget_data=scenario_data['forget_data'],
            retain_data=scenario_data['retain_data'],
            device=device,
            variance_threshold=0.85,
            force_recompute=True,
            cache_dir=f'./test_cache_{scenario_name.replace(" ", "_")}'
        )
        
        # Check success criteria
        if 'forget_comparison' in results and 'retain_comparison' in results:
            forget_sims = [c['mean_cosine'] for c in results['forget_comparison'].values() if c]
            retain_sims = [c['mean_cosine'] for c in results['retain_comparison'].values() if c]
            
            if forget_sims and retain_sims:
                mean_forget_sim = np.mean(forget_sims)
                mean_retain_sim = np.mean(retain_sims)
                
                print(f"    Forget similarity: {mean_forget_sim:.3f}")
                print(f"    Retain similarity: {mean_retain_sim:.3f}")
                
                # Success if forget similarity is high and retain similarity is low
                success = mean_forget_sim > 0.7 and mean_retain_sim < 0.5
                return success
        
        return False
        
    except Exception as e:
        print(f"    Error testing scenario: {e}")
        return False