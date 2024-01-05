import numpy as np
import os
import tensorflow as tf

#Generar dataset para X con 150 valores
num_values = 150
input_values = np.linspace(-10.0, 10.0, num_values)

#fórmula asignada
output_values = -99 * input_values + 40 + np.random.normal(0, 5, len(input_values))

#Entrenar epochs asignados que son 600
tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
])

linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mean_squared_error)
print(linear_model.summary())

num_epochs = 600
linear_model.fit(input_values, output_values, epochs=num_epochs)

#Probar el modelo con predict para 14 valores asignados
test_input_values = np.linspace(-10.0, 10.0, 14).reshape((-1, 1))
predictions = linear_model.predict(test_input_values).flatten()
print("Predictions:", predictions)

#Exportar el modelo con el nombre asignado en modelname
model_name = 'modelec2'
export_path = f'./{model_name}/1/'
tf.saved_model.save(linear_model, os.path.join('./', export_path))

#Extraer los pesos para W y b e imprimirlos
weights, biases = linear_model.layers[0].get_weights()
print(f"(W): {weights.flatten()[0]}")
print(f"(b): {biases[0]}")
