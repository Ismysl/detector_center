import cv2
import numpy as np

cap = cv2.VideoCapture("Uaz2.mov")
_, frame = cap.read()


# Получение разрешения кадров
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Определение кодек и создание объекта VideoWriter. Результат будет сохранен в файле 'output.avi'.
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Кодек безопасный вариант
out = cv2.VideoWriter('output.avi', fourcc, 15, (frame_width, frame_height))

bbox = cv2.selectROI(frame, True, False)

cv2.destroyAllWindows()

tracker = cv2.legacy.TrackerCSRT_create()
status_tracker = tracker.init(frame, bbox)
fps = 0

while True:
    status_cap, frame = cap.read()
    
    # Параметры цветового фильтра
    low_threshold = np.array([0,90,0], dtype=np.uint8)
    high_threshold = np.array([170,255,255], dtype=np.uint8)
    mask = cv2.inRange(frame, low_threshold, high_threshold)
    image_new = cv2.bitwise_and(frame, frame, mask=mask)

    # Преобразование в оттенки серого
    gray = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (3, 3), 5)
    edge = cv2.Canny(gaussian, 10, 150)
    
    # Поиск контуров
    contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not status_cap:
        break

    if status_tracker:
        timer = cv2.getTickCount()
        status_tracker, bbox = tracker.update(frame)

    if status_tracker:
        x, y, w, h = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Вычисляем координаты центра квадрата
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
	
	
	# Перебираем все контуры
        inside_contour = False  # Флаг для отслеживания, была ли найдена точка внутри контура
        for contour in contours:
            if cv2.pointPolygonTest(contour, (center_x, center_y), False) > 0:  # Проверяем, находится ли точка клика внутри контура
                # Если да, то отрисовываем этот контур
                #cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                
                # Рассчитываем и отображаем центр контура
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Рисуем желтое перекрестие в центре контура
                    cv2.drawMarker(frame, (cX, cY), (0, 255, 255), cv2.MARKER_CROSS, thickness=2)
                    #print(f"Center coordinates: ({cX}, {cY})")
                    inside_contour = True  # Устанавливаем флаг в True, если точка внутри контура
                break  # Прекращаем цикл после нахождения первого подходящего контура
        
        # Если точка не была найдена внутри контура, ищем ближайший контур
        if not inside_contour:
            radius = 10  # Задаем желаемый радиус в пикселях
        
            # Находим расстояния от точки клика до всех контуров
            distances = [(np.linalg.norm(np.array(contour[0]) - np.array([center_x, center_y]))) for contour in contours]
        
            # Ищем индексы контуров, находящихся в заданном радиусе
            close_contours_indices = [i for i, d in enumerate(distances) if d <= radius]
        
            if close_contours_indices:
                # Среди подходящих контуров находим самый большой по площади        
                areas = [cv2.contourArea(contours[i]) for i in close_contours_indices]
                max_area_index = np.argmax(areas)
        
                # Отрисовываем найденный контур
                closest_contour_index = close_contours_indices[max_area_index]
                #cv2.drawContours(image, [contours[closest_contour_index]], -1, (0, 255, 0), 2)

                # Рассчитываем и отображаем центр контура
                M = cv2.moments(contours[closest_contour_index])
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Рисуем желтое перекрестие в центре контура
                    cv2.drawMarker(frame, (cX, cY), (0, 255, 255), cv2.MARKER_CROSS, thickness=2)
                    #print(f"Center coordinates: ({cX}, {cY})")


        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        cv2.putText(frame, "FPS: %.0f" % fps, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3);
    else:
        cv2.putText(frame, "Tracking failure detected", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("MedianFlow tracker", frame)
    
    # Запись обработанного кадра обратно в видео файл
    out.write(frame)

    k = cv2.waitKey(10)

    if k == 27: 
        break

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()
