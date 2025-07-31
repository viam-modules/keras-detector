# Module keras-detector 

Viam provides a `keras-detector` model of the [vision service](/services/vision) with which you can use to interface with detectors made with Keras.

### Configuration
The following attribute template can be used to configure this model:

```json
{
"model_path": <string>,
"camera_name": <string>
}
```

#### Attributes

The following attributes are available for this model:

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `model_path`  | string | Required  | The filepath to your Keras detector file.  Should end with '.keras' |
| `camera_name` | string | Required  | The name of the camera configured on your robot. |

#### Example Configuration

```json
{
  "model_path": "/Users/ViamUser/models/myDetector.keras",
  "camera_name": "camera-1"
}
```
