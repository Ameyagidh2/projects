/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package library.management;
import java.awt.*;
import javax.swing.*;
import java.sql.*;
import java.awt.event.*;
import javax.swing.border.*;

/**
 *
 * @author ameya
 */
public class AboutUs extends JFrame {
    private JLabel j1,j2,j3,j4,j5,j6,j7,j8;
    private JPanel controlPanel;
    
    public static void main(String[] args) {
        new AboutUs().setVisible(true);
    }
    
    public AboutUs(){
        super("About Us");
        setLayout(null);
        setBackground(Color.WHITE);
        setBounds(500,250,600,500);
        
        controlPanel = new JPanel(); 
        controlPanel.setLayout(null);
        setContentPane(controlPanel);
        
        j1 = new JLabel("Library" );
        j1.setBounds(160,40,150,55);
        j1.setFont(new Font("Trebuchet Ms",Font.BOLD,35));
        j1.setForeground(Color.GREEN);
        controlPanel.add(j1);
        
        // long words in 1 line increase the width length of the line
        j1 = new JLabel("Management System" );
        j1.setBounds(110,90,405,55);
        j1.setFont(new Font("Trebuchet Ms",Font.BOLD,35));
        j1.setForeground(Color.CYAN);
        controlPanel.add(j1);
        
        j1 = new JLabel("V.1.0" );
        j1.setBounds(160,120,470,55);
        j1.setFont(new Font("Trebuchet Ms",Font.BOLD,35));
        j1.setForeground(Color.GREEN);
        
        controlPanel.add(j1);
        j1 = new JLabel("Developed by:-" );
        j1.setBounds(10,150,450,55);
        j1.setFont(new Font("Trebuchet Ms",Font.BOLD,15));
        j1.setForeground(Color.BLACK);
        controlPanel.add(j1);
        controlPanel.add(j1);
        
        j1 = new JLabel("Ameya S Gidh" );
        j1.setBounds(220,150,450,55);
        j1.setFont(new Font("Trebuchet Ms",Font.BOLD,15));
        j1.setForeground(Color.GREEN);
        controlPanel.add(j1);
        
        j1 = new JLabel("Contact:-" );
        j1.setBounds(10,250,450,55);
        j1.setFont(new Font("Trebuchet Ms",Font.BOLD,15));
        j1.setForeground(Color.BLACK);
        controlPanel.add(j1);
        controlPanel.add(j1);
        
        j1 = new JLabel("+1 8574379760" );
        j1.setBounds(220,250,450,55);
        j1.setFont(new Font("Trebuchet Ms",Font.BOLD,15));
        j1.setForeground(Color.GREEN);
        controlPanel.add(j1);
        
        j1 = new JLabel("email id " );
        j1.setBounds(10,350,450,55);
        j1.setFont(new Font("Trebuchet Ms",Font.BOLD,15));
        j1.setForeground(Color.BLACK);
        controlPanel.add(j1);
        controlPanel.add(j1);
        
        j1 = new JLabel("ameyagidh2@gmail.com" );
        j1.setBounds(170,350,450,55);
        j1.setFont(new Font("Trebuchet Ms",Font.BOLD,15));
        j1.setForeground(Color.GREEN);
        controlPanel.add(j1);
        
    }

}

