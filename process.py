import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt


def apply_fourier_transform(image):  
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    # Magnitud del espectro
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    return magnitude_spectrum
    

def azimuthal_average(image, center=None):
    # Calcular el perfil radial de la imagen
    height, width = image.shape
    y, x = np.indices(image.shape)

    # Si no se especifica el centro, se calcula como el centro de la imagen
    if center is None:
        center = np.array([height/2.0, width/2.0])

    # Calcular las distancias de cada píxel al centro
    distances = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    distances = distances.flatten()
    pixels = image.flatten()

    # Ordenar por distancia al centro
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_pix = pixels[sorted_indices]

    # Convertir distancias a enteros para agrupar
    int_distances  = sorted_distances.astype(int)
    changes = np.diff(int_distances)
    group_int = np.where(changes != 0)[0]

    # Cantidad de píxeles por anillo radial
    cant = np.diff(group_int, prepend=0)
    
   # Sumar los valores de píxeles por anillo
    cum_sum = np.cumsum(sorted_pix, dtype=float)
    ring_sum = np.diff(cum_sum[group_int], prepend=0)

    # Calcular el promedio por anillo
    radial_profile = ring_sum / cant

    return radial_profile

def save_results(radial_profile, filename, features_dir):
    # Guardar el perfil radial
    features_path = os.path.join(features_dir, f"radial_{filename}.npy")
    radial_profile = np.nan_to_num(radial_profile, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(features_path, radial_profile)

    print(f"Imagen {filename} procesada con éxito")

def crop_image(image, crop_pixels):
    # Obtener dimensiones originales
    width, height = image.shape[:2]
    # Recortar (izquierda, arriba, derecha, abajo)
    cropped_image = image[crop_pixels:height - crop_pixels, crop_pixels:width - crop_pixels]

    return cropped_image

        
def proccess_images(input_dir, features_dir, crop_pixels):
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    
    # Procesar cada imagen en el directorio de entrada
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            
            # Leer la imagen en escala de grises
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error al cargar la imagen {filename}")
                continue
            
            # Redimensionar la imagen a 800x800 píxeles
            cropped_image = crop_image(image, crop_pixels)
             
            # Aplicar la Transformada de Fourier y obtener el espectro de magnitud
            magnitude_spectrum = apply_fourier_transform(cropped_image)
            #radial_profile = azimuthal_average(magnitude_spectrum)
            radial_profile = azimuthal_average(magnitude_spectrum)

            save_results(radial_profile, filename, features_dir)



if __name__ == "__main__":
    # Directorio con las imágenes de entrada
    input_dir = ".../imagenes/200/real"  
    # Directorio para guardar los resultados
    features_dir = ".../features/200/real"  
   
    crop_pixels = 20 
    
    proccess_images(input_dir, features_dir, crop_pixels)


    


            

    
      



    
