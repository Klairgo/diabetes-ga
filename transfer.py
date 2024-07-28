import mendeley as m
import pima as p
import iammus_transfer as i
import random as rand


if __name__ == '__main__':

    rand.seed(989)
    print("Running mendeley and saving to file...")
    men = m.gp()
    men.write_to_file()

    print("Running pima and saving to file...")
    pim = p.gp()
    pim.write_to_file()

    print("Running iammus...")
    iammus = i.gp()
    i.run()

