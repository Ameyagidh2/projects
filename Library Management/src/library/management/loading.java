/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package library.management;

/**
 *
 * @author ameya
 */
import javax.swing.*;
import java.awt.*;
import java.sql.*;
import java.awt.event.*;

public class loading extends JFrame implements Runnable{
    
    private JProgressBar progressBar;
    private JPanel contentPanel;
    int s = 0;
    Thread th;
    Connection conn;
    
    // constructor
    // multiple inheritance possible in java by interface    
    public static void main(String[] args)
    {
        new loading().setVisible(true);
    }
     public void setUploading()
    {
        th.start();
        // start method will call other method internally
    }
    
    public void run()
    {
    try
    {
      for(int i= 0; i < 200; i ++)
      {
          s = s + 1;
          int m = progressBar.getMaximum();
          int v = progressBar.getValue();
          if (v < m)
          {
             progressBar.setValue(progressBar.getValue() + 1);
             
          }
          else
          {
              i = 201;
              setVisible(false);
              new Home().setVisible(true);
          }
          Thread.sleep(20);
      }
    }
    catch(Exception e){
        e.printStackTrace();
    
    }
    }
    
    public loading()
    {   
        super("Loading"); // sper always on top its heading
        // for threading use a runnable and this call
        th = new Thread((Runnable) this);
        setBounds(600,300,600,400);
        contentPanel = new JPanel();
        setContentPane(contentPanel);
        contentPanel.setLayout(null);
       
        JLabel load = new JLabel("Smart Library");
        load.setBounds(130,46,500,35);
        load.setForeground(new Color(72,209,204));
        load.setFont(new Font("Trebuchet MS", Font.BOLD, 35));
        contentPanel.add(load);
        
        progressBar = new JProgressBar();
        progressBar.setBackground(new Color(72, 209, 204));
        load.setFont(new Font("Trebuchet MS", Font.BOLD, 20));
        progressBar.setStringPainted(true);
        progressBar.setBounds(130,135,300,25);
        contentPanel.add(progressBar);
        
         JLabel j2 = new JLabel("Please Wait...");
         j2.setBounds(200,165,150,20);
         j2.setForeground(Color.red);
         contentPanel.add(j2);
         
          JPanel panel = new JPanel();
          panel.setBackground(Color.WHITE);
          panel.setBounds(0, 0, 590, 390);
          contentPanel.add(panel);
        
          setUploading();
    }
    

}
