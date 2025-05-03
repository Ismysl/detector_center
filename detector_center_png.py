import cv2
import numpy as np

# Чтение изображения
image = cv2.imread('6.png')

# Параметры цветового фильтра
low_threshold = np.array([0,90,0], dtype=np.uint8)
high_threshold = np.array([170,255,255], dtype=np.uint8)
mask = cv2.inRange(image, low_threshold, high_threshold)
image_new = cv2.bitwise_and(image, image, mask=mask)

# Преобразование в оттенки серого
gray = cv2.cvtColor(image_new, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(gray, (3, 3), 5)
edge = cv2.Canny(gaussian, 10, 150)
cv2.imshow('Edge', edge)

# Поиск контуров
contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def mouse_click(event, x, y, flags, param):
    global image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Перебираем все контуры
        inside_contour = False  # Флаг для отслеживания, была ли найдена точка внутри контура
        for contour in contours:
            if cv2.pointPolygonTest(contour, (x, y), False) > 0:  # Проверяем, находится ли точка клика внутри контура
                # Если да, то отрисовываем этот контур
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                
                # Рассчитываем и отображаем центр контура
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Рисуем желтое перекрестие в центре контура
                cv2.drawMarker(image, (cX, cY), (0, 255, 255), cv2.MARKER_CROSS, thickness=2)
                print(f"Center coordinates: ({cX}, {cY})")
                inside_contour = True  # Устанавливаем флаг в True, если точка внутри контура
                break  # Прекращаем цикл после нахождения первого подходящего контура
        
        # Если точка не была найдена внутри контура, ищем ближайший контур
        if not inside_contour:
            radius = 10  # Задаем желаемый радиус в пикселях
        
            # Находим расстояния от точки клика до всех контуров
            distances = [(np.linalg.norm(np.array(contour[0]) - np.array([x, y]))) for contour in contours]
        
            # Ищем индексы контуров, находящихся в заданном радиусе
            close_contours_indices = [i for i, d in enumerate(distances) if d <= radius]
        
            if close_contours_indices:
                # Среди подходящих контуров находим самый большой по площади        
                areas = [cv2.contourArea(contours[i]) for i in close_contours_indices]
                max_area_index = np.argmax(areas)
        
                # Отрисовываем найденный контур
                closest_contour_index = close_contours_indices[max_area_index]
                cv2.drawContours(image, [contours[closest_contour_index]], -1, (0, 255, 0), 2)

                # Рассчитываем и отображаем центр контура
                M = cv2.moments(contours[closest_contour_index])
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Рисуем желтое перекрестие в центре контура
                cv2.drawMarker(image, (cX, cY), (0, 255, 255), cv2.MARKER_CROSS, thickness=2)
                print(f"Center coordinates: ({cX}, {cY})")

# Создание окна и привязка функции обработки кликов
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_click, param=image_new)

while True:
    cv2.imshow('Image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

