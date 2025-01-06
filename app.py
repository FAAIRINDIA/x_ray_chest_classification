import streamlit as st
import numpy as np
import altair as alt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

def compute_gradcam(model, img):
    try:
        # Ensure img is a valid tensor with correct shape
        if not isinstance(img, tf.Tensor):
            return None
            
        if len(img.shape) != 4:
            return None

        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
                
        if last_conv_layer is None:
            print("Could not find convolutional layer")
            return None

        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )

        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Get gradients
        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            print("Gradient computation failed")
            return None

        # Global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels by corresponding gradients
        conv_outputs = conv_outputs.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        # Generate heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Get original image
        original_img = img[0].numpy()
        # Normalize to 0-255 range
        original_img = ((original_img - original_img.min()) * 255 / 
                       (original_img.max() - original_img.min() + 1e-10))
        original_img = original_img.astype(np.uint8)
        
        # Ensure original image is RGB
        if original_img.shape[-1] == 1:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        
        # Create the superimposed image
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        return superimposed_img

    except Exception as e:
        print(f"Error in compute_gradcam: {str(e)}")
        return None

def process_image(img_path):
    try:
        # Load and preprocess the image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        
        # Normalize the image
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        return img_tensor

    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return None

# Load the model
@st.cache_resource
def load_model():
    try:
        base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(14, activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights("pretrained_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Rest of your Streamlit app code
model = load_model()
if model is None:
    st.error("Failed to load model")
    st.stop()

st.success("Model loaded successfully!")

# Labels for classification
labels = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

st.title("Medical Image Classification with Grad-CAM")
st.write("Upload an X-ray image to classify it and visualize the model's focus using Grad-CAM.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Save and process the image
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process image for model input
        img_tensor = process_image(file_path)
        if img_tensor is None:
            st.error("Failed to process image")
            st.stop()

        # Make predictions
        predictions = model.predict(img_tensor)[0]

        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            'Class': labels,
            'Probability': predictions
        })

        # Create and display chart
        chart = alt.Chart(prediction_df).mark_bar().encode(
            x=alt.X('Probability', title='Prediction Probability'),
            y=alt.Y('Class', sort='-x', title='Condition'),
            color='Class'
        ).properties(
            title='Prediction Confidence per Class'
        )
        st.altair_chart(chart, use_container_width=True)

        # Generate and display Grad-CAM
        grad_cam_img = compute_gradcam(model, img_tensor)
        if grad_cam_img is not None:
            st.image(grad_cam_img, caption="Grad-CAM Heatmap", use_column_width=True)
        else:
            st.error("Failed to generate Grad-CAM visualization")

        # Clean up
        os.remove(file_path)
        
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
########################################################################################################
########################################################################################################
# import streamlit as st
# import numpy as np
# import altair as alt
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import os
# import pandas as pd
# import cv2
# import tensorflow as tf
# import matplotlib.pyplot as plt

# # Grad-CAM function
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow.keras.preprocessing import image as keras_image


# def compute_gradcam(model, img):
#     try:
#         # Check if img is None
#         if img is None:
#             print("Error: The image is None.")
#             return None

#         # Prepare the model for Grad-CAM (use the last convolutional layer)
#         last_conv_layer = model.get_layer('conv5_block16_2_conv')  # For DenseNet121, check your model layer names
        
#         # Create a model that outputs the activations of the last convolutional layer and the model's predictions
#         grad_model = tf.keras.models.Model(
#             inputs=[model.inputs],
#             outputs=[last_conv_layer.output, model.output]
#         )

#         # Compute the gradient of the predicted class w.r.t the output of the last convolutional layer
#         with tf.GradientTape() as tape:
#             tape.watch(img)
#             conv_outputs, predictions = grad_model(img)
#             pred_index = np.argmax(predictions[0])  # Get the index of the predicted class
#             output = predictions[:, pred_index]
#             grads = tape.gradient(output, conv_outputs)

#         # Pool the gradients across all the axes leaving out the channel dimension
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#         # Extract the output of the last convolutional layer
#         conv_outputs = conv_outputs[0]  # Convert to numpy array
#         conv_outputs = conv_outputs.numpy()

#         # Multiply the output feature map by the pooled gradients
#         for i in range(conv_outputs.shape[-1]):
#             conv_outputs[:, :, i] *= pooled_grads[i]

#         # The resulting feature map is the weighted sum of all the channels
#         heatmap = np.mean(conv_outputs, axis=-1)

#         # Normalize the heatmap
#         heatmap = np.maximum(heatmap, 0)
#         heatmap = cv2.resize(heatmap, (224, 224))  # Resize heatmap to match input image size
#         heatmap /= np.max(heatmap)  # Normalize the heatmap to [0, 1]

#         # Convert heatmap to RGB (color map)
#         heatmap = np.uint8(255 * heatmap)
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#         # Check if original image is valid (convert if necessary)
#         original_img = img.numpy() if isinstance(img, tf.Tensor) else img
#         if original_img is None or original_img.shape[0] == 0 or original_img.shape[1] == 0:
#             print("Error: The original image is empty or invalid.")
#             return None

#         # If image has 1 channel (grayscale), convert to RGB
#         if len(original_img.shape) == 3 and original_img.shape[-1] == 1:
#             original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
#         elif len(original_img.shape) == 3 and original_img.shape[-1] != 3:
#             original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

#         # Resize original image to match heatmap
#         original_img_resized = cv2.resize(original_img, (224, 224))
        
#         # Superimpose the heatmap onto the original image
#         superimposed_img = cv2.addWeighted(original_img_resized, 0.6, heatmap, 0.4, 0)

#         return superimposed_img

#     except Exception as e:
#         # Catching the error to understand where the issue is coming from
#         print(f"Error processing the file: {str(e)}")
#         return None

# def process_image(img_path):
#     try:
#         # Load and process the image
#         img = keras_image.load_img(img_path, target_size=(224, 224))
#         img_array = keras_image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)

#         # Normalize the image
#         img_array = tf.image.per_image_standardization(img_array)

#         return img_array

#     except Exception as e:
#         print(f"Error processing the image: {str(e)}")
#         return None

# # Load the model
# @st.cache_resource
# def load_model():
#     base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     predictions = Dense(14, activation="sigmoid")(x)  # Adjust to the number of labels
#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.load_weights("pretrained_model.h5")  # Replace with your weights file
#     return model

# model = load_model()
# st.success("Model loaded successfully!")

# # Labels for classification
# labels = [
#     "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
#     "Effusion", "Emphysema", "Fibrosis", "Hernia",
#     "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
#     "Pneumonia", "Pneumothorax"
# ]

# # Streamlit app layout
# st.title("Medical Image Classification with Grad-CAM")
# st.write("Upload an X-ray image to classify it and visualize the model's focus using Grad-CAM.")

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
#     # Preprocess the image
#     try:
#         file_path = os.path.join("uploads", uploaded_file.name)
#         os.makedirs("uploads", exist_ok=True)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         img = load_img(file_path, target_size=(224, 224))  # Adjust input size if necessary
#         img_array = img_to_array(img)
#         img_array_for_gradcam = np.copy(img_array)  # We need the original image for Grad-CAM
#         img_array = preprocess_input(img_array)
#         img_array = np.expand_dims(img_array, axis=0)

#         # Convert to tf.Tensor
#         img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

#         # Make predictions
#         predictions = model.predict(img_array)[0]

#         # Create a dataframe for the predictions
#         prediction_df = pd.DataFrame({
#             'Class': labels,
#             'Probability': predictions
#         })

#         # Create Altair bar chart
#         chart = alt.Chart(prediction_df).mark_bar().encode(
#             x=alt.X('Probability', title='Prediction Probability'),
#             y=alt.Y('Class', sort='-x', title='Condition'),
#             color='Class'
#         ).properties(
#             title='Prediction Confidence per Class'
#         )

#         # Display the chart
#         st.altair_chart(chart, use_container_width=True)

#         # Grad-CAM computation
#         grad_cam_img = compute_gradcam(model, img_tensor)

#         # Display the Grad-CAM image
#         st.image(grad_cam_img, caption="Grad-CAM Heatmap", use_column_width=True)

#         # Clean up
#         os.remove(file_path)
#     except Exception as e:
#         st.error(f"Error processing the file: {e}")



########################################################################################
########################################################################################

# import streamlit as st
# import numpy as np
# import altair as alt
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import os
# import pandas as pd

# # Load the model
# @st.cache_resource
# def load_model():
#     base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     predictions = Dense(14, activation="sigmoid")(x)  # Adjust to the number of labels
#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.load_weights("pretrained_model.h5")  # Replace with your weights file
#     return model

# model = load_model()
# st.success("Model loaded successfully!")

# # Labels for classification
# labels = [
#     "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
#     "Effusion", "Emphysema", "Fibrosis", "Hernia",
#     "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
#     "Pneumonia", "Pneumothorax"
# ]

# # Streamlit app layout
# st.title("Medical Image Classification")
# st.write("Upload an X-ray image to classify it into the following conditions:")
# st.write(", ".join(labels))

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
#     # Preprocess the image
#     try:
#         file_path = os.path.join("uploads", uploaded_file.name)
#         os.makedirs("uploads", exist_ok=True)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         img = load_img(file_path, target_size=(224, 224))  # Adjust input size if necessary
#         img_array = img_to_array(img)
#         img_array = preprocess_input(img_array)
#         img_array = np.expand_dims(img_array, axis=0)

#         # Make predictions
#         predictions = model.predict(img_array)[0]

#         # Create a dataframe for the predictions
#         prediction_df = pd.DataFrame({
#             'Class': labels,
#             'Probability': predictions
#         })

#         # Create Altair bar chart
#         chart = alt.Chart(prediction_df).mark_bar().encode(
#             x=alt.X('Probability', title='Prediction Probability'),
#             y=alt.Y('Class', sort='-x', title='Condition'),
#             color='Class'
#         ).properties(
#             title='Prediction Confidence per Class'
#         )

#         # Display the chart
#         st.altair_chart(chart, use_container_width=True)

#         # Clean up
#         os.remove(file_path)
#     except Exception as e:
#         st.error(f"Error processing the file: {e}")

##############################################################################################
# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import os
# import matplotlib.pyplot as plt

# # Load the model
# @st.cache_resource
# def load_model():
#     base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     predictions = Dense(14, activation="sigmoid")(x)  # Adjust to the number of labels
#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.load_weights("pretrained_model.h5")  # Replace with your weights file
#     return model

# model = load_model()
# st.success("Model loaded successfully!")

# # Labels for classification
# labels = [
#     "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
#     "Effusion", "Emphysema", "Fibrosis", "Hernia",
#     "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
#     "Pneumonia", "Pneumothorax"
# ]

# # Streamlit app layout
# st.title("Medical Image Classification")
# st.write("Upload an X-ray image to classify it into the following conditions:")
# st.write(", ".join(labels))

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
#     # Preprocess the image
#     try:
#         file_path = os.path.join("uploads", uploaded_file.name)
#         os.makedirs("uploads", exist_ok=True)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         img = load_img(file_path, target_size=(224, 224))  # Adjust input size if necessary
#         img_array = img_to_array(img)
#         img_array = preprocess_input(img_array)
#         img_array = np.expand_dims(img_array, axis=0)

#         # Make predictions
#         predictions = model.predict(img_array)[0]
        
#         # Display results
#         st.write("### Prediction Results:")
#         for label, score in zip(labels, predictions):
#             st.write(f"{label}: {score:.4f}")

#         # Plotting the predictions as a bar chart
#         fig, ax = plt.subplots()
#         ax.barh(labels, predictions, color='skyblue')
#         ax.set_xlabel('Prediction Probability')
#         ax.set_title('Prediction Confidence per Class')
#         st.pyplot(fig)

#         # Clean up
#         os.remove(file_path)
#     except Exception as e:
#         st.error(f"Error processing the file: {e}")



# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import os

# # Load the model
# @st.cache_resource
# def load_model():
#     base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     predictions = Dense(14, activation="sigmoid")(x)  # Adjust to the number of labels
#     model = Model(inputs=base_model.input, outputs=predictions)
#     model.load_weights("pretrained_model.h5")  # Replace with your weights file
#     return model

# model = load_model()
# st.success("Model loaded successfully!")

# # Labels for classification
# labels = [
#     "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
#     "Effusion", "Emphysema", "Fibrosis", "Hernia",
#     "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
#     "Pneumonia", "Pneumothorax"
# ]

# # Streamlit app layout
# st.title("Medical Image Classification")
# st.write("Upload an X-ray image to classify it into the following conditions:")
# st.write(", ".join(labels))

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
#     # Preprocess the image
#     try:
#         file_path = os.path.join("uploads", uploaded_file.name)
#         os.makedirs("uploads", exist_ok=True)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         img = load_img(file_path, target_size=(224, 224))  # Adjust input size if necessary
#         img_array = img_to_array(img)
#         img_array = preprocess_input(img_array)
#         img_array = np.expand_dims(img_array, axis=0)

#         # Make predictions
#         predictions = model.predict(img_array)[0]
        
#         # Display results
#         st.write("### Prediction Results:")
#         for label, score in zip(labels, predictions):
#             st.write(f"{label}: {score:.4f}")

#         # Clean up
#         os.remove(file_path)
#     except Exception as e:
#         st.error(f"Error processing the file: {e}")
