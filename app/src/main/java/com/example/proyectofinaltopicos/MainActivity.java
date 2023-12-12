package com.example.proyectofinaltopicos;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

// Clase principal de la actividad
public class MainActivity extends AppCompatActivity {
    // Constantes para la captura de imagen y permisos
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int PERMISSIONS_REQUEST_CODE = 100;

    // Vistas de la interfaz de usuario
    private ImageView imageView;
    private Interpreter tflite;

    // Bitmap para almacenar la imagen actual
    private Bitmap currentImageBitmap;

    // Método de creación de la actividad
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Establece el layout de la actividad
        setContentView(R.layout.activity_main);

        // Inicializa las vistas de la interfaz de usuario
        imageView = findViewById(R.id.imageView);
        Button takePhotoButton = findViewById(R.id.buttonTakePhoto);
        Button evaluateButton = findViewById(R.id.buttonEvaluate);

        // Establece los oyentes para los botones
        takePhotoButton.setOnClickListener(v -> openCamera());
        evaluateButton.setOnClickListener(v -> evaluateModel());

        // Carga el modelo de TensorFlow Lite
        loadModel();
    }

    // Método para abrir la cámara
    private void openCamera() {
        // Verifica los permisos de la cámara importante ya que segun documentacion en las versiones de android actuales es necesario
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            // Solicita los permisos si no están concedidos
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PERMISSIONS_REQUEST_CODE);
        } else {
            // Inicia la intención para tomar una foto
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            }
        }
    }

    // Un dolor de cabeza :c
    // Método para evaluar el modelo con la imagen actual
    private void evaluateModel() {
        if (currentImageBitmap != null) {
            // Procesa la imagen para el modelo y obtiene los resultados
            Bitmap processedImage = processImageForModel(currentImageBitmap);
            float[] results = runModel(processedImage);
            // Muestra los resultados
            showResults(results);
        }
    }

    // Procesa la imagen para adaptarla al modelo
    private Bitmap processImageForModel(Bitmap bitmap) {
        // Redimensiona y convierte la imagen a escala de grises
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 20, 20, true);
        Bitmap grayscaleBitmap = Bitmap.createBitmap(20, 20, Bitmap.Config.ARGB_8888);

        // Convierte cada píxel a escala de grises
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 20; j++) {
                int pixel = resizedBitmap.getPixel(i, j);
                int gray = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3;
                grayscaleBitmap.setPixel(i, j, Color.rgb(gray, gray, gray));
            }
        }

        return grayscaleBitmap;
    }

    // La imagen la trabajamos en funcion a la resolucion establecida en el modelo en collab
    // Ejecuta el modelo con la imagen procesada
    private float[] runModel(Bitmap bitmap) {
        // Convierte la imagen a ByteBuffer para el modelo
        ByteBuffer inputBuffer = convertBitmapToByteBuffer(bitmap);
        // Asigna espacio para la salida del modelo (asumiendo 20 clases)
        float[][] outputArray = new float[1][20];
        // Ejecuta el modelo
        tflite.run(inputBuffer, outputArray);
        return outputArray[0];
    }

    // Convierte la imagen de Bitmap a ByteBuffer
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(20 * 20 * 4);
        byteBuffer.order(ByteOrder.nativeOrder());
        // Obtiene los píxeles de la imagen
        int[] pixels = new int[20 * 20];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // Convierte cada píxel a escala de grises y lo añade al buffer
        for (int pixel : pixels) {
            float grayscaleValue = ((Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3.0f) / 255.0f;
            byteBuffer.putFloat(grayscaleValue);
        }

        return byteBuffer;
    }

    // Carga el modelo de TensorFlow Lite desde los activos
    private void loadModel() {
        try {
            MappedByteBuffer buffer = loadModelFile();
            // Crea el intérprete de TensorFlow Lite con el modelo cargado
            tflite = new Interpreter(buffer);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Carga el archivo del modelo desde los activos
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("modelo_convertido.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        // Mapea el modelo en memoria para ser leído
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Método llamado al regresar de la actividad de la cámara
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // Verifica si la imagen fue capturada con éxito
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK) {
            Bundle extras = data.getExtras();
            // Obtiene la imagen capturada y la muestra en el ImageView
            currentImageBitmap = (Bitmap) extras.get("data");
            imageView.setImageBitmap(currentImageBitmap);
        }
    }

    // Muestra los resultados de la predicción
    private void showResults(float[] prediction) {
        // Obtiene el índice de la clase con la mayor probabilidad
        int labelIndex = getIndexOfLargest(prediction);
        // Obtiene el nombre de la etiqueta correspondiente
        String labelName = getLabelName(labelIndex);

        // Encuentra el TextView en la interfaz de usuario y actualiza su texto
        TextView resultView = findViewById(R.id.textViewResult);
        resultView.setText(labelName);
    }

    // Obtiene el índice del elemento más grande en un arreglo
    private int getIndexOfLargest(float[] array) {
        if (array == null || array.length == 0) return -1;
        int largestIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[largestIndex]) largestIndex = i;
        }
        return largestIndex;
    }

    // Devuelve el nombre de la etiqueta correspondiente al índice
    private String getLabelName(int index) {
        // Lista de nombres de etiquetas (ajustar según tu modelo)
        String[] labelNames = {
                "Ninguna","Lupino Azul", "Viguiera", "Dodonaea viscosa", "Trompetero amarillo",
                "Rabano Salvaje", "Narciso Peruano", "Chirimoya", "Aguaymanto", "Capuli",
                "Ciruela", "Durazno", "Flor Campanilla Fucsia", "Granadilla", "Limon",
                "Manzana", "Naranja", "Papa", "Romero", "Rosa", "Ruda "
        };
        return labelNames[index];
    }
}