from pyorbbecsdk import *
import cv2
import os

def list_devices():
    ctx = Context()
    device_list = ctx.query_devices()
    print(f"Nombre de caméras détectées : {device_list.get_count()}")

    for i in range(device_list.get_count()):
        device = device_list[i]
        device_info = device.get_device_info()
        print(f"Caméra {i}: {device_info}")

    return device_list

def test_device(device):
    config = Config()
    pipeline = Pipeline(device)
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        print("Profils disponibles :")
        for profile in profile_list:
            print(profile)

        try:
            color_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError:
            color_profile = profile_list.get_default_video_stream_profile()
            print("Profil par défaut utilisé :", color_profile)

        config.enable_stream(color_profile)
        pipeline.start(config)

        frames = pipeline.wait_for_frames(100)
        color_frame = frames.get_color_frame()
        if color_frame is None:
            print("Pas de frame couleur.")
            return False

        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            print("Impossible de convertir en image.")
            return False

        cv2.imshow("Test Caméra", color_image)
        print("Appuyez sur une touche pour continuer...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True

    except Exception as e:
        print("Erreur avec cette caméra :", e)
        return False
    finally:
        pipeline.stop()

def main():
    device_list = list_devices()
    print("\n--- Test des caméras ---")
    return
    for i in range(device_list.get_count()):
        print(f"\n--- Test de la caméra #{i} ---")
        device = device_list.get_device(i)
        success = test_device(device)
        if success:
            print(f"✅ Caméra #{i} fonctionne.")
        else:
            print(f"❌ Caméra #{i} ne donne pas d'image couleur.")

if __name__ == "__main__":
    main()