def calculate_estimation(masked_pixels, image_width_pixels, door_height_pixels=None, door_height_ft=7.0, material_type='paint'):
    """
    Calculates surface area, required material, and estimated cost.
    """
    # Determine the scale (sq ft per pixel)
    if door_height_pixels and door_height_pixels > 0:
        feet_per_pixel = door_height_ft / door_height_pixels
        sq_ft_per_pixel = feet_per_pixel ** 2
    else:
        # Fallback heuristic: assume house is 30 ft wide
        feet_per_pixel = 30.0 / image_width_pixels
        sq_ft_per_pixel = feet_per_pixel ** 2

    # Calculate actual area
    total_area_sq_ft = masked_pixels * sq_ft_per_pixel
    total_area_sq_ft = round(total_area_sq_ft, 2)
    
    # Wastage factor
    WASTAGE_PERCENT = 0.10
    total_material_needed = total_area_sq_ft * (1 + WASTAGE_PERCENT)

    # Rates DB ($ per sq ft)
    rates = {
        'paint': {'material': 0.50, 'labor': 1.00, 'unit': 'gallons', 'coverage_per_unit': 350},
        'tiles': {'material': 5.00, 'labor': 4.00, 'unit': 'tiles', 'coverage_per_unit': 1.0}, # Assume 1x1 ft tile
        'stone': {'material': 12.00, 'labor': 8.00, 'unit': 'sq ft', 'coverage_per_unit': 1.0},
        'texture': {'material': 2.00, 'labor': 2.50, 'unit': 'bags', 'coverage_per_unit': 50.0}
    }

    if material_type not in rates:
        material_type = 'paint' # fallback

    rate_info = rates[material_type]
    
    # Calculate costs
    material_cost = total_material_needed * rate_info['material']
    labor_cost = total_material_needed * rate_info['labor']
    total_cost = material_cost + labor_cost
    
    # Calculate quantities
    units_needed = total_material_needed / rate_info['coverage_per_unit']

    return {
        'surface_area_sq_ft': total_area_sq_ft,
        'wastage_percent': int(WASTAGE_PERCENT * 100),
        'material_quantity_needed': round(total_material_needed, 2),
        'units_needed': round(units_needed, 1),
        'unit_type': rate_info['unit'],
        'costs': {
            'material_rate_sqft': rate_info['material'],
            'labor_rate_sqft': rate_info['labor'],
            'material_total': round(material_cost, 2),
            'labor_total': round(labor_cost, 2),
            'grand_total': round(total_cost, 2)
        }
    }
