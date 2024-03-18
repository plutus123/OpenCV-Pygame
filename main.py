import sys, time, random, pygame
from collections import deque
import cv2 as cv, mediapipe as mp


pygame.display.set_caption(' ')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
pygame.init()


VID_CAP = cv.VideoCapture(0)
window_size = (int(VID_CAP.get(cv.CAP_PROP_FRAME_WIDTH)), int(VID_CAP.get(cv.CAP_PROP_FRAME_HEIGHT)))
screen = pygame.display.set_mode(window_size)


bird_img = pygame.image.load("bird_sprite.png")
bird_img = pygame.transform.scale(bird_img, (bird_img.get_width() / 6, bird_img.get_height() / 6))
bird_frame = bird_img.get_rect()


pipe_img = pygame.image.load("pipe_sprite_single.png")
pipe_starting_template = pipe_img.get_rect()
space_between_pipes = 250


time_between_pipe_spawn = 40
dist_between_pipes = 500
pipe_velocity = lambda: dist_between_pipes / time_between_pipe_spawn
level = 0
score = 0
didUpdateScore = False

def initialize_game():
    global bird_frame, pipe_frames, score, game_is_running, game_over, stage, pipeSpawnTimer, game_clock
    bird_frame.center = (window_size[0] // 6, window_size[1] // 2)
    pipe_frames = deque()
    score = 0
    game_is_running = True
    game_over = False
    stage = 1
    pipeSpawnTimer = 0
    game_clock = time.time()
initialize_game()

def spawn_pipe():
    top = pipe_starting_template.copy()
    top.x, top.y = window_size[0], random.randint(120 - 1000, window_size[1] - 120 - space_between_pipes - 1000)
    bottom = pipe_starting_template.copy()
    bottom.x, bottom.y = window_size[0], top.y + 1000 + space_between_pipes
    pipe_frames.append([top, bottom])

def update_pipe_positions_and_check_collisions():
    global score, didUpdateScore, pipeSpawnTimer, game_is_running
    for pf in pipe_frames:
        pf[0].x -= pipe_velocity()
        pf[1].x -= pipe_velocity()
    if len(pipe_frames) > 0 and pipe_frames[0][0].right < 0:
        pipe_frames.popleft()
    checker = True
    for pf in pipe_frames:
        if pf[0].left <= bird_frame.x <= pf[0].right:
            checker = False
            if not didUpdateScore:
                score += 1
                didUpdateScore = True
        screen.blit(pipe_img, pf[1])
        screen.blit(pygame.transform.flip(pipe_img, False, True), pf[0])
    if checker:
        didUpdateScore = False

    #  score text
    # text = pygame.font.SysFont("Helvetica Bold.ttf", 50).render(f'Stage {stage}', True, (99, 245, 255))
    # tr = text.get_rect()
    # tr.center = (100, 50)
    # screen.blit(text, tr)
    text = pygame.font.SysFont("Helvetica Bold.ttf", 50).render(f'Score: {score}', True, (99, 245, 255))
    tr = text.get_rect()
    tr.center = (100, 100)
    screen.blit(text, tr)


    if any(bird_frame.colliderect(pf[0]) or bird_frame.colliderect(pf[1]) for pf in pipe_frames):
        game_is_running = False
    if pipeSpawnTimer == 0:
        spawn_pipe()
    pipeSpawnTimer = (pipeSpawnTimer + 1) % time_between_pipe_spawn

def display_game_over_screen():
    text = pygame.font.SysFont("Helvetica Bold.ttf", 64).render('Game Over! Press Enter to restart.', True, (99, 245, 255))
    tr = text.get_rect(center=(window_size[0] / 2, window_size[1] / 2))
    screen.blit(text, tr)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RETURN]:
        initialize_game()




with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                VID_CAP.release()
                cv.destroyAllWindows()
                pygame.quit()
                sys.exit()

        if game_is_running:
            ret, frame = VID_CAP.read()
            if not ret:
                print("Empty frame, continuing...")
                continue
            frame.flags.writeable = False
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            if results.multi_face_landmarks:
                marker = results.multi_face_landmarks[0].landmark[94].y
                bird_frame.centery = int((marker - 0.5) * 1.5 * window_size[1] + window_size[1] / 2)
                if bird_frame.top < 0: bird_frame.y = 0
                if bird_frame.bottom > window_size[1]: bird_frame.y = window_size[1] - bird_frame.height
            frame = cv.flip(frame, 1).swapaxes(0, 1)
            screen.fill((255, 255, 255))
            pygame.surfarray.blit_array(screen, frame)
            screen.blit(bird_img, bird_frame)
            update_pipe_positions_and_check_collisions()

        else:  # Game Over screen
            display_game_over_screen()

        pygame.display.flip()
