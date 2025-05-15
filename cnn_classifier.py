import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np

class WasteClassifier(nn.Module):
    def __init__(self, num_classes=6):  # Default to 6 classes
        super(WasteClassifier, self).__init__()
        # Use MobileNetV2 as the base model
        self.base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze the base model parameters initially
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Replace the classifier head with a more flexible architecture
        self.base_model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base_model.last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
    
    def unfreeze_layers(self, num_layers=10):
        """Unfreeze the last few layers for fine-tuning"""
        for param in list(self.base_model.parameters())[-num_layers:]:
            param.requires_grad = True
            
    def get_transform(self, train=True):
        """Get the appropriate transforms for training or validation"""
        if train:
            return transforms.Compose([
                transforms.Resize((224, 224)),  # Standard size for MobileNetV2
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])

class CNNWasteClassifier:
    def __init__(self, model_path='waste_cnn.pt', device=None):
        self.device = device or torch.device('cpu')
        
        # List of allowed waste categories
        self.allowed_categories = [
            'plastic', 'paper', 'organic', 'foodpackaging', 'glass', 'teabag',
            'aerosol_cans', 'aluminum_soda_cans', 'cardboard_boxes', 
            'cardboard_packaging', 'coffee_grounds', 'eggshells', 
            'shoes', 'styrofoam_cups'
        ]
        
        # Map model class names to internal names
        self.class_mapping = {
            'teabags': 'teabag'  # Map plural form from model to singular form in code
        }
        
        # Confidence threshold for classification
        self.confidence_threshold = 0.15  # Lower threshold to accept more predictions
        
        try:
            # Try to load the saved model state with classes
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint and 'classes' in checkpoint:
                # Initialize model with the correct number of classes from saved model
                self.model = WasteClassifier(num_classes=len(checkpoint['classes']))
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Map model classes to internal representation
                self.classes = [self.class_mapping.get(cls, cls) for cls in checkpoint['classes']]
                
                # Verify all classes are in allowed categories
                for cls in self.classes:
                    mapped_cls = self.class_mapping.get(cls, cls)
                    if mapped_cls not in self.allowed_categories:
                        print(f"Warning: Found unsupported category '{cls}' in model")
            else:
                # Fallback to default with all 14 classes if model doesn't have class info
                self.model = WasteClassifier(num_classes=len(self.allowed_categories))
                self.model.load_state_dict(checkpoint)
                self.classes = self.allowed_categories
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create new model if loading fails with all 14 classes
            self.model = WasteClassifier(num_classes=len(self.allowed_categories))
            self.classes = self.allowed_categories
            
        # Set model to evaluation mode and move to device
        self.model.eval().to(self.device)
        
        # Get the transforms for inference
        self.tfms = self.model.get_transform(train=False)

    def classify(self, image: Image.Image):
        """Classify a single image"""
        img = self.tfms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # Apply specific enhancements for certain categories based on image characteristics
        # Simple heuristic checks to improve category detection
        
        # Check if image is likely to be shoes based on color and shape patterns
        is_likely_shoes = self._check_for_shoes(image)
        if is_likely_shoes:
            # Find the index for shoes
            shoe_index = -1
            for i, class_name in enumerate(self.classes):
                if class_name == 'shoes':
                    shoe_index = i
                    break
                    
            if shoe_index >= 0:
                # Increase probability for shoes
                probs[shoe_index] *= 2.0  # Stronger boost for shoes
                
        # Apply uniform confidence boost to all categories
        boost_factor = 1.5  # Higher boost factor for stronger confidence
        # Apply boost to all waste categories
        for i in range(len(self.classes)):
            probs[i] *= boost_factor
        
        # Renormalize after boosting
        probs = probs / np.sum(probs)
        
        # Get top 3 predictions (or fewer if we have fewer classes)
        num_predictions = min(3, len(self.classes))
        top_indices = probs.argsort()[-num_predictions:][::-1]
        predictions = []
        
        for idx in top_indices:
            class_name = self.classes[idx]
            mapped_class = self.class_mapping.get(class_name, class_name)
            # Only include predictions above threshold
            if probs[idx] >= self.confidence_threshold:
                predictions.append({
                    'type': mapped_class,
                    'confidence': float(probs[idx])
                })
        
        return predictions
        
    def _check_for_shoes(self, image):
        """Detect if image is likely to contain shoes based on simple heuristics"""
        # Convert to smaller size for faster processing
        small_img = image.resize((100, 100))
        
        # Convert to numpy array and get metrics
        img_array = np.array(small_img)
        
        # Check for shoe-like characteristics
        # Shoes typically have specific color distributions and shapes
        
        # 1. Color diversity check - shoes often have variety of colors
        if len(img_array.shape) == 3:  # Check if color image
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            r_std = np.std(r)
            g_std = np.std(g)
            b_std = np.std(b)
            
            # Check for moderate color variation (too high might be something else)
            color_variation = (r_std + g_std + b_std) / 3
            if 20 < color_variation < 80:
                return True
                
        # 2. Check for shoe-like aspect ratio
        width, height = image.size
        aspect_ratio = width / height
        
        # Shoes typically have a horizontal orientation
        if 1.2 < aspect_ratio < 2.5:
            return True
            
        return False

    def get_class_info(self, class_name):
        """Get information about a specific waste class"""
        waste_info = {
            'plastic': {
                'recycling_steps': [
                    'Clean and dry the plastic item',
                    'Remove any labels or caps',
                    'Check the recycling number',
                    'Place in appropriate recycling bin'
                ],
                'environmental_impact': {
                    'decomposition_time': '450 years',
                    'recyclability': 'High',
                    'energy_saved': '75%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '85 kWh',
                    'water_per_kg': '180 liters',
                    'co2_per_kg': '2.5 kg',
                    'avg_weight': {
                        'bottle': 0.025,
                        'bag': 0.01,
                        'container': 0.05,
                        'default': 0.03
                    }
                }
            },
            'paper': {
                'recycling_steps': [
                    'Remove any non-paper items',
                    'Flatten cardboard boxes',
                    'Keep paper dry and clean',
                    'Place in paper recycling bin'
                ],
                'environmental_impact': {
                    'decomposition_time': '2-6 weeks',
                    'recyclability': 'Very High',
                    'energy_saved': '65%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '4 kWh',
                    'water_per_kg': '50 liters',
                    'co2_per_kg': '1.1 kg',
                    'avg_weight': {
                        'newspaper': 0.2,
                        'cardboard': 0.25,
                        'office_paper': 0.075,
                        'default': 0.1
                    }
                }
            },
            'organic': {
                'recycling_steps': [
                    'Separate from other waste',
                    'Compost at home if possible',
                    'Use municipal composting if available',
                    'Avoid mixing with non-organic waste'
                ],
                'environmental_impact': {
                    'decomposition_time': '2-4 weeks',
                    'recyclability': '100%',
                    'energy_saved': '90%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '0.8 kWh',
                    'water_per_kg': '5 liters',
                    'co2_per_kg': '0.5 kg',
                    'avg_weight': {
                        'food_scraps': 0.2,
                        'leaves': 0.05,
                        'garden_waste': 0.5,
                        'default': 0.25
                    }
                }
            },
            'aerosol_cans': {
                'recycling_steps': [
                    'Completely empty the can',
                    'Do not puncture or crush',
                    'Remove plastic caps',
                    'Place in metal recycling bin'
                ],
                'environmental_impact': {
                    'decomposition_time': '200–500 years',
                    'recyclability': 'High',
                    'energy_saved': '95%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '12 kWh',
                    'water_per_kg': '25 liters',
                    'co2_per_kg': '1.6 kg',
                    'avg_weight': {
                        'small_can': 0.2,
                        'large_can': 0.5,
                        'default': 0.3
                    }
                }
            },
            'aluminum_soda_cans': {
                'recycling_steps': [
                    'Rinse out remaining liquid',
                    'Crush if required by local program',
                    'Place in aluminum recycling bin'
                ],
                'environmental_impact': {
                    'decomposition_time': '80–200 years',
                    'recyclability': 'Very High',
                    'energy_saved': '92%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '8.5 kWh',
                    'water_per_kg': '20 liters',
                    'co2_per_kg': '1.4 kg',
                    'avg_weight': {
                        'standard_can': 0.015,
                        'tall_can': 0.02,
                        'default': 0.015
                    }
                }
            },
            'cardboard_boxes': {
                'recycling_steps': [
                    'Flatten boxes',
                    'Remove any plastic tape or labels',
                    'Bundle or place loose in cardboard bin'
                ],
                'environmental_impact': {
                    'decomposition_time': '2 months',
                    'recyclability': 'High',
                    'energy_saved': '24%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '1.5 kWh',
                    'water_per_kg': '100 liters',
                    'co2_per_kg': '0.6 kg',
                    'avg_weight': {
                        'small_box': 0.5,
                        'medium_box': 1.0,
                        'large_box': 2.0,
                        'default': 1.0
                    }
                }
            },
            'cardboard_packaging': {
                'recycling_steps': [
                    'Remove plastic windows or liners',
                    'Flatten and stack',
                    'Place in paper/cardboard recycling'
                ],
                'environmental_impact': {
                    'decomposition_time': '2 months',
                    'recyclability': 'High',
                    'energy_saved': '22%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '1.6 kWh',
                    'water_per_kg': '95 liters',
                    'co2_per_kg': '0.5 kg',
                    'avg_weight': {
                        'small_package': 0.1,
                        'medium_package': 0.3,
                        'default': 0.2
                    }
                }
            },
            'coffee_grounds': {
                'recycling_steps': [
                    'Let grounds cool',
                    'Collect in paper filter or compostable bag',
                    'Add to compost or municipal organics'
                ],
                'environmental_impact': {
                    'decomposition_time': '2–6 months',
                    'recyclability': 'Compostable',
                    'energy_saved': '10%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '0.2 kWh',
                    'water_per_kg': '15 liters',
                    'co2_per_kg': '0.3 kg',
                    'avg_weight': {
                        'per_cup': 0.02,
                        'per_batch': 0.1,
                        'default': 0.05
                    }
                }
            },
            'eggshells': {
                'recycling_steps': [
                    'Rinse to remove membrane',
                    'Crush into small pieces',
                    'Add to compost or garden soil'
                ],
                'environmental_impact': {
                    'decomposition_time': '3–6 months',
                    'recyclability': 'Compostable',
                    'energy_saved': '5%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '0.1 kWh',
                    'water_per_kg': '5 liters',
                    'co2_per_kg': '0.1 kg',
                    'avg_weight': {
                        'single_shell': 0.01,
                        'dozen': 0.12,
                        'default': 0.05
                    }
                }
            },
            'shoes': {
                'recycling_steps': [
                    'Clean off excess dirt',
                    'Remove laces and insoles',
                    'Donate reusable pairs or send to shoe-recycler'
                ],
                'environmental_impact': {
                    'decomposition_time': '25–50 years (leather), 75–100+ years (synthetic)',
                    'recyclability': 'Variable',
                    'energy_saved': '18%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '3.0 kWh',
                    'water_per_kg': '50 liters',
                    'co2_per_kg': '2.0 kg',
                    'avg_weight': {
                        'sneaker': 0.8,
                        'boot': 1.5,
                        'sandal': 0.3,
                        'default': 0.9
                    }
                }
            },
            'styrofoam_cups': {
                'recycling_steps': [
                    'Remove liquid residue',
                    'Check for local EPS collection',
                    'If not recyclable locally, discard in trash'
                ],
                'environmental_impact': {
                    'decomposition_time': '500+ years',
                    'recyclability': 'Low',
                    'energy_saved': '2%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '0.8 kWh',
                    'water_per_kg': '30 liters',
                    'co2_per_kg': '0.4 kg',
                    'avg_weight': {
                        'small_cup': 0.01,
                        'large_cup': 0.02,
                        'default': 0.015
                    }
                }
            },
            'foodpackaging': {
                'recycling_steps': [
                    'Rinse food residue from packaging',
                    'Separate different materials (plastic, paper, foil)',
                    'Check local recycling guidelines',
                    'Place in appropriate recycling bins'
                ],
                'environmental_impact': {
                    'decomposition_time': '100+ years',
                    'recyclability': 'Medium',
                    'energy_saved': '60%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '65 kWh',
                    'water_per_kg': '160 liters',
                    'co2_per_kg': '2.2 kg',
                    'avg_weight': {
                        'takeout_container': 0.03,
                        'fast_food_package': 0.025,
                        'wrapper': 0.01,
                        'default': 0.02
                    }
                }
            },
            'glass': {
                'recycling_steps': [
                    'Rinse glass containers',
                    'Remove caps and lids',
                    'Separate by color if required',
                    'Place in glass recycling bin'
                ],
                'environmental_impact': {
                    'decomposition_time': '1 million+ years',
                    'recyclability': 'High',
                    'energy_saved': '30%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '4.5 kWh',
                    'water_per_kg': '40 liters',
                    'co2_per_kg': '0.9 kg',
                    'avg_weight': {
                        'bottle': 0.3,
                        'jar': 0.2,
                        'cup': 0.15,
                        'default': 0.25
                    }
                }
            },
            'teabag': {
                'recycling_steps': [
                    'Empty used tea leaves into compost',
                    'Check if the teabag is compostable',
                    'Remove any staples or tags',
                    'Place compostable teabags in food waste'
                ],
                'environmental_impact': {
                    'decomposition_time': '6 months',
                    'recyclability': 'Medium',
                    'energy_saved': '50%'
                },
                'conservation_metrics': {
                    'energy_per_kg': '1.2 kWh',
                    'water_per_kg': '15 liters',
                    'co2_per_kg': '0.4 kg',
                    'avg_weight': {
                        'teabag': 0.003,
                        'tea_waste': 0.005,
                        'packaging': 0.01,
                        'default': 0.005
                    }
                }
            }
        }
        
        # Return generic info if category not recognized
        if class_name not in waste_info:
            return {
                'recycling_steps': ['Check local recycling guidelines for this type of waste'],
                'environmental_impact': {
                    'decomposition_time': 'Varies',
                    'recyclability': 'Check local guidelines',
                    'energy_saved': 'Varies'
                },
                'conservation_metrics': {
                    'energy_per_kg': 'Unknown',
                    'water_per_kg': 'Unknown',
                    'co2_per_kg': 'Unknown',
                    'avg_weight': {
                        'default': 0.1
                    }
                }
            }
        
        return waste_info[class_name]
