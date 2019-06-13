import time
from django.test import TestCase
from django.test import LiveServerTestCase, Client
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from cancerApp.models import Doctor


# Create your tests here.
class STestClass(LiveServerTestCase):
    def setUp(self):

        self.selenium = webdriver.Chrome(
            executable_path="/home/abanoub/chromedriver_linux64/chromedriver"
        )
        self.selenium.get("http://127.0.0.1:8000/")
        self.selenium.maximize_window()
        super(STestClass, self).setUp()

    def tearDown(self):
        self.selenium.quit()
        super(STestClass, self).tearDown()

    def testLoginSuccess(self):
        selenium = self.selenium
        # Opening the link we want to test
        # find the form element
        selenium.find_element_by_id('login-btn').click()
        username = selenium.find_element_by_id('id_username')
        password = selenium.find_element_by_id('id_password')
        submit = selenium.find_element_by_id('login_submit')
    
        # Fill the form with data
        username.send_keys('admin')
        password.send_keys('password')
        # submitting the form
        submit.send_keys(Keys.RETURN)
        # check the returned result
        assert selenium.find_element_by_id('logout-btn')
    
    def testLoginFail(self):
        selenium = self.selenium
        # Opening the link we want to test
        # find the form element
        selenium.find_element_by_id('login-btn').click()
        username = selenium.find_element_by_id('id_username')
        password = selenium.find_element_by_id('id_password')
        submit = selenium.find_element_by_id('login_submit')
    
        # Fill the form with data
        username.send_keys('')
        password.send_keys('password')
        # submitting the form
        submit.send_keys(Keys.RETURN)
    
        # check the returned result
        assert submit

    def test_prediction(self):
        selenium = self.selenium
        selenium.find_element_by_id('login-btn').click()
        username = selenium.find_element_by_id('id_username')
        password = selenium.find_element_by_id('id_password')
        submit = selenium.find_element_by_id('login_submit')
    
        # Fill the form with data
        username.send_keys('admin')
        password.send_keys('password')
        # submitting the form
        submit.send_keys(Keys.RETURN)
        # check the returned result
        dropdownlist = selenium.find_element_by_id('dropdown04')
        ActionChains(selenium).move_to_element(dropdownlist).perform()
        selenium.find_element_by_id('cancer-btn').click()
        selenium.find_element_by_id('imageUpload').send_keys('/home/abanoub/LIDC-IDRI-0001.zip')
        selenium.find_element_by_id('btn-predict').click()
        time.sleep(5)
        self.assertEquals(selenium.find_element_by_xpath("//*[@id='diagnose_img']").get_attribute('src'),
                          'http://127.0.0.1:8000/static/cancerApp/img/unhealthy.jpg')

    def test_rib_visualization(self):
        selenium = self.selenium
        selenium.find_element_by_id('login-btn').click()
        username = selenium.find_element_by_id('id_username')
        password = selenium.find_element_by_id('id_password')
        submit = selenium.find_element_by_id('login_submit')

        # Fill the form with data
        username.send_keys('admin')
        password.send_keys('password')
        # submitting the form
        submit.send_keys(Keys.RETURN)
        # check the returned result
        dropdownlist = selenium.find_element_by_id('dropdown04')
        ActionChains(selenium).move_to_element(dropdownlist).perform()
        selenium.find_element_by_id('vis-btn').click()
        selenium.find_element_by_id('imageUpload').send_keys('/home/abanoub/LIDC-IDRI-0001.zip')
        selenium.find_element_by_id('visualizeRib').click()
        selenium.find_element_by_id('btn-visualize').click()
        time.sleep(200)
        resulting_img = str(selenium.find_element_by_id("lung_img").get_attribute('src'))
        self.assertTrue('http://127.0.0.1:8000/static/cancerApp/img/lungfig.jpg' in resulting_img)

    def test_lung_visualization(self):
        selenium = self.selenium
        selenium.find_element_by_id("login-btn").click()
        username = selenium.find_element_by_id("id_username")
        password = selenium.find_element_by_id("id_password")
        submit = selenium.find_element_by_id("login_submit")

        # Fill the form with data
        username.send_keys("admin")
        password.send_keys("password")
        # submitting the form
        submit.send_keys(Keys.RETURN)
        # check the returned result
        dropdownlist = selenium.find_element_by_id("dropdown04")
        ActionChains(selenium).move_to_element(dropdownlist).perform()
        selenium.find_element_by_id("vis-btn").click()
        selenium.find_element_by_id("imageUpload").send_keys(
            "/home/abanoub/LIDC-IDRI-0001.zip"
        )
        selenium.find_element_by_id("visualizeLung").click()
        selenium.find_element_by_id("btn-visualize").click()
        time.sleep(150)
        resulting_img = str(
            selenium.find_element_by_id("lung_img").get_attribute("src")
        )
        self.assertTrue(
            "http://127.0.0.1:8000/static/cancerApp/img/lungfig.jpg" in resulting_img
        )

    def test_cancerSpread(self):
        selenium = self.selenium
        selenium.find_element_by_id("login-btn").click()
        username = selenium.find_element_by_id("id_username")
        password = selenium.find_element_by_id("id_password")
        submit = selenium.find_element_by_id("login_submit")

        # Fill the form with data
        username.send_keys("admin")
        password.send_keys("password")
        # submitting the form
        submit.send_keys(Keys.RETURN)
        # check the returned result
        dropdownlist = selenium.find_element_by_id("dropdown04")
        ActionChains(selenium).move_to_element(dropdownlist).perform()
        selenium.find_element_by_id("vis-btn").click()
        selenium.find_element_by_id("imageUpload").send_keys(
            "/home/abanoub/LIDC-IDRI-0001.zip"
        )
        selenium.find_element_by_id("cancerSpread").click()
        selenium.find_element_by_id("btn-visualize").click()
        time.sleep(60)
        resulting_img = str(
            selenium.find_element_by_id("lung_img").get_attribute("src")
        )
        self.assertTrue(
            "http://127.0.0.1:8000/static/cancerApp/img/lungfig.jpg" in resulting_img
        )

    def test_showStats(self):
        selenium = self.selenium
        selenium.find_element_by_id("login-btn").click()
        username = selenium.find_element_by_id("id_username")
        password = selenium.find_element_by_id("id_password")
        submit = selenium.find_element_by_id("login_submit")

        # Fill the form with data
        username.send_keys("admin")
        password.send_keys("password")
        # submitting the form
        submit.send_keys(Keys.RETURN)
        # check the returned result
        dropdownlist = selenium.find_element_by_id("dropdown04")
        ActionChains(selenium).move_to_element(dropdownlist).perform()
        selenium.find_element_by_id("stats-btn").click()
        selenium.find_element_by_id('cancer').click()
        time.sleep(2)
        selenium.find_element_by_id('gender').click()
        time.sleep(2)
        selenium.find_element_by_id('age').click()
        time.sleep(10)
        imgs = selenium.find_elements_by_css_selector('img')
        self.assertTrue(len(imgs) == 3)

    # not tested yet
    def test_clearHistory(self):
    	selenium = self.selenium
        selenium.find_element_by_id("login-btn").click()
        username = selenium.find_element_by_id("id_username")
        password = selenium.find_element_by_id("id_password")
        submit = selenium.find_element_by_id("login_submit")

        # Fill the form with data
        username.send_keys("admin")
        password.send_keys("password")
        # submitting the form
        submit.send_keys(Keys.RETURN)
        # check the returned result
        dropdownlist = selenium.find_element_by_id("dropdown04")
        ActionChains(selenium).move_to_element(dropdownlist).perform()
        selenium.find_element_by_id("stats-btn").click()
        selenium.find_element_by_id('h_clear').click()


