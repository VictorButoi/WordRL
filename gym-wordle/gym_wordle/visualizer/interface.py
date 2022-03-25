import pygame

pygame.init()

white = (255, 255, 255)
black = (0, 0, 0)
grey = (69, 69, 69)

# FROM OFFICIAL WORDLE GAME
green = (120, 177, 90)
yellow = (253, 203, 88)

# GAME SETTINGS
NUM_ROWS = 6
NUM_COLUMNS = 5

# GRAPHICS CONSTANTS, add 100 for spacing
BOX_SIZE = 100
BOX_SPACING = 10
SCREEN_SPACING = 200
SCREEN_HEIGHT = NUM_ROWS*BOX_SIZE + (NUM_ROWS - 1)*BOX_SPACING + SCREEN_SPACING
SCREEN_WIDTH = NUM_COLUMNS*BOX_SIZE + \
    (NUM_COLUMNS - 1)*BOX_SPACING + SCREEN_SPACING

# PYGAME OVERHEAD
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
pygame.display.set_caption("Wordle Knockoff")
turn = 0
fps = 60
timer = pygame.time.Clock()
huge_font = pygame.font.Font("freesansbold.ttf", 56)
font_x = 40
font_y = 48
game_over = False
secret_word = 'power'

# GRAPHICS SETUP
board = [[" " for _ in range(NUM_COLUMNS)] for _ in range(NUM_ROWS)]

# TODO: remove idx


def place_letter(x, y, x_idx, box_size):
    wordle_list = ["A", "B", "C", "D", "E"]

    piece_text = huge_font.render(wordle_list[x_idx], True, white)
    x_offset = (box_size - font_x)/2
    y_offset = (box_size - font_y)/2
    screen.blit(piece_text, (x + x_offset, y + y_offset))


def draw_board(board,
               sox,
               soy,
               box_size,
               x_space,
               y_space,
               do_fill=0):
    height = len(board)
    width = len(board[0])
    wordle_list = ["W", "R", "D", "R", "L"]
    colors = [green, yellow, grey, green, yellow, grey]

    for row in range(height):
        for col in range(width):
            # convention for drawing is [x, y, width, height], boarder, rounding
            x = col*box_size + sox + col*x_space
            y = row*box_size + soy + row*y_space
            pygame.draw.rect(screen, colors[row], [
                             x, y, box_size, box_size], do_fill)
            place_letter(x, y, col, box_size)


x_offset = (SCREEN_WIDTH - (NUM_COLUMNS*BOX_SIZE +
            (NUM_COLUMNS - 1)*BOX_SPACING))/2
y_offset = (SCREEN_HEIGHT - (NUM_ROWS*BOX_SIZE + (NUM_ROWS - 1)*BOX_SPACING))/2

running = True
while running:
    timer.tick(fps)
    screen.fill(black)
    draw_board(board, sox=x_offset, soy=y_offset, box_size=BOX_SIZE,
               x_space=BOX_SPACING, y_space=BOX_SPACING)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.TEXTINPUT and not game_over:
            entry = event.__getattribute__('text')

    pygame.display.flip()
